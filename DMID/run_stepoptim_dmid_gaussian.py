import argparse
import glob
import os
import PIL
from PIL import Image
from matplotlib import pyplot as plt
from natsort import natsorted
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import lpips

from utils.utils_image import tensor2uint, calculate_psnr, calculate_ssim
from utils.utils_sampling import generalized_steps
from step_optim import NoiseScheduleVP, StepOptim
from guided_diffusion.unet import UNetModel
from Data import Dataset
import csv


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timestamps):
    if beta_schedule == "linear": return np.linspace(beta_start, beta_end, num_diffusion_timestamps, dtype=np.float64)
    raise NotImplementedError(beta_schedule)


def data_transform(X): return 2 * X - 1.0


def data_transform_reverse(X): return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DenoisingEngine(object):

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device_id'])
        self.model = UNetModel()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(config['save']['ddpm_checkpoint'], map_location=self.device))
        self.model.eval()
        if torch.cuda.device_count() > 1: self.model = nn.DataParallel(self.model)
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        betas = get_beta_schedule(beta_schedule=config['diffusion']['beta_schedule'],
                                  beta_start=config['diffusion']['beta_start'],
                                  beta_end=config['diffusion']['beta_end'],
                                  num_diffusion_timestamps=config['diffusion']['num_diffusion_timestamps'])
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.alphas_cumprod = (1.0 - self.betas).cumprod(dim=0)
        self.ns = NoiseScheduleVP(schedule='discrete', alphas_cumprod=self.alphas_cumprod)
        self.step_optim = StepOptim(self.ns)
        # Cache for generated sampling sequences
        self.sequence_cache = {}
        print('____________________________ Denoising Engine Initialized ____________________________')

    def sample_image(self, x_noisy, T_start, custom_seq, eta=0.85):
        # Renosing step for Gaussian noise removal
        t = torch.full((x_noisy.shape[0],), T_start - 1, device=self.device, dtype=torch.long)
        alpha_t_cumprod = self.alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)

        x_N = x_noisy * alpha_t_cumprod.sqrt()

        # Use the renosed x_N as input to the sampler
        xs = generalized_steps(x=x_N, seq=custom_seq, model=self.model, b=self.betas, eta=eta)
        return xs[0][-1]

    def run_single_evaluation(self, dataloader, T_start, nfe, R_t, schedule_type='original'):
        results = {'psnr': [], 'ssim': [], 'lpips': []}

        cache_key = (T_start, nfe, schedule_type)
        if cache_key in self.sequence_cache:
            sequence = self.sequence_cache[cache_key]
        else:
            if schedule_type == 'original':
                # Baseline: uniform/linear step sequence
                skip = T_start // nfe if nfe > 0 else 1
                seq = list(range(0, T_start, skip))
                sequence = np.array(seq[:nfe] + [T_start - 1])
                sequence = np.unique(sequence)
            else:  # optimized
                # Generate optimized sequence using StepOptim
                print(f"Generating new optimized sequence for T_start={T_start}, NFE={nfe}...")
                self.ns.T = T_start / 1000.0
                self.step_optim.T = T_start / 1000.0
                t_steps, _ = self.step_optim.get_ts_lambdas(N=nfe, eps=1 / 1000.0, initType='unif_t')

                # Remove duplicate steps that StepOptim might generate
                sequence = np.unique(np.flip((t_steps.numpy() * 1000).astype(int)))

                print("--- Generated Optimized Sequence ---")
                print(sequence)
                print("-" * 34)

            self.sequence_cache[cache_key] = sequence

        # The actual NFE might be different due to `np.unique`
        actual_nfe = len(sequence) - 1 if len(sequence) > 0 else 0

        with torch.no_grad():
            for index, (noisy_tensor, clean_tensor) in enumerate(
                    tqdm(dataloader, desc=f"T={T_start}, NFE={nfe}, type={schedule_type}")):
                noisy_img, clean_img = noisy_tensor.to(self.device), clean_tensor.to(self.device)
                h, w = noisy_img.shape[2], noisy_img.shape[3]
                # Pad image dimensions to be divisible by 64 (for U-Net)
                factor, padh, padw = 64, (64 - h % 64) % 64, (64 - w % 64) % 64
                noise_img_padded = F.pad(noisy_img, (0, padw, 0, padh), 'reflect')

                x_noisy_transformed = data_transform(noise_img_padded)

                denoised_sum = torch.zeros_like(x_noisy_transformed)
                # Perform ensembling R_t times
                for _ in range(R_t):
                    denoised_sum += self.sample_image(x_noisy_transformed, T_start, sequence, eta=0.85).to(self.device)
                denoised_avg = denoised_sum / R_t
                denoised_avg = torch.nan_to_num(denoised_avg, nan=0.0, posinf=1.0, neginf=-1.0)
                denoised_img = data_transform_reverse(denoised_avg)

                restored_np = tensor2uint(denoised_img[:, :, :h, :w])
                clean_np = tensor2uint(clean_img)
                results['psnr'].append(calculate_psnr(restored_np, clean_np, border=0))
                results['ssim'].append(calculate_ssim(restored_np, clean_np, border=0))
                results['lpips'].append(
                    self.lpips_fn(data_transform(denoised_img[:, :, :h, :w]), data_transform(clean_img)).item())

        return {metric: np.mean(values) for metric, values in results.items()}, actual_nfe


class ExperimentRunner:

    def __init__(self, config, save_path):
        self.engine = DenoisingEngine(config)
        self.all_results = {}
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        # T_start (N) lookup table from the paper
        self.n_list = [33, 57, 115, 215, 291, 348, 393]
        self.sigma_list = [15, 25, 50, 100, 150, 200, 250]

        # S_t (NFE) lookup table from the paper
        self.s_t_params = {
            'CBSD68': {15: 13, 25: 23, 50: 28, 100: 43, 150: 72, 200: 69, 250: 98},
            'Kodak': {15: 23, 25: 43, 50: 57, 100: 107, 150: 97, 200: 86, 250: 98},
            'McMaster': {15: 23, 25: 43, 50: 57, 100: 107, 150: 143, 200: 173, 250: 196},
            'Imagenet': {15: 23, 25: 43, 50: 57, 100: 107, 150: 145, 200: 173, 250: 196}  # Using Kodak as proxy
        }

    def get_t_start(self, sigma):
        # Find the closest sigma in the list to get the corresponding T_start (N)
        closest_sigma = min(self.sigma_list, key=lambda x: abs(x - sigma))
        idx = self.sigma_list.index(closest_sigma)
        return self.n_list[idx]

    def run_all_experiments(self):
        datasets_to_run = ['CBSD68', 'Kodak', 'McMaster', 'Imagenet']
        sigmas_to_run = [50, 100, 150, 200, 250]
        # Set ensembling runs to 1 for all experiments
        R_t = 1

        for name in datasets_to_run:
            self.all_results[name] = {}
            for sigma in sigmas_to_run:
                print(f"\n{'=' * 40}\nStarting Experiment for: {name} (Sigma={sigma})\n{'=' * 40}")
                self.all_results[name][sigma] = {'Original Paper': {}, 'Optimized': {}}

                data_path = f'./data/{name}'
                if not os.path.exists(data_path):
                    print(f"Warning: Dataset path not found: {data_path}. Skipping.")
                    continue

                dataset = Dataset(path_clean=data_path,
                                  path_noise=data_path,  # Set same as path_clean
                                  noise_sigma=sigma,
                                  opt=name)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

                # Get baseline NFE (S_t) from the lookup table
                T_start = self.get_t_start(sigma)
                S_t_orig = self.s_t_params[name][sigma]

                # --- Run Baseline ---
                orig_results, nfe_orig_actual = self.engine.run_single_evaluation(dataloader, T_start, S_t_orig, R_t,
                                                                                  schedule_type='original')
                self.all_results[name][sigma]['Original Paper'] = {'NFE': nfe_orig_actual, 'metrics': orig_results}

                # --- Generate NFE list for optimized method ---
                nfe_exp_list = [S_t_orig]
                if S_t_orig < 50:
                    step_size = 5
                elif S_t_orig < 100:
                    step_size = 10
                else:
                    step_size = 20
                for i in range(1, 6):
                    next_nfe = S_t_orig - (step_size * i)
                    if next_nfe > 0: nfe_exp_list.append(next_nfe)

                # --- Run Optimized Experiments ---
                for nfe_exp in nfe_exp_list:
                    exp_results, nfe_exp_actual = self.engine.run_single_evaluation(dataloader, T_start, nfe_exp, R_t,
                                                                                    schedule_type='optimized')

                    # Use actual NFE as the key
                    self.all_results[name][sigma]['Optimized'][f'NFE={nfe_exp_actual}'] = exp_results

        print("\nAll experiments finished.")

    def save_results_to_csv(self, filename="summary_results.csv"):
        save_filepath = os.path.join(self.save_path, filename)
        print(f"\nSaving all results to {save_filepath}...")

        header = ['Dataset', 'Sigma', 'Method', 'NFE', 'PSNR', 'SSIM', 'LPIPS']

        with open(save_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)  # Write CSV header

            for dataset_name, sigma_data in self.all_results.items():
                for sigma, data in sigma_data.items():
                    # Write baseline result
                    orig_data = data['Original Paper']
                    orig_nfe = orig_data['NFE']
                    orig_metrics = orig_data['metrics']
                    row_orig = [dataset_name, sigma, 'Original Paper', orig_nfe,
                                f"{orig_metrics['psnr']:.4f}", f"{orig_metrics['ssim']:.4f}",
                                f"{orig_metrics['lpips']:.4f}"]
                    writer.writerow(row_orig)

                    # Write optimized results, sorted by NFE
                    sorted_optimized = sorted(data['Optimized'].items(), key=lambda item: int(item[0].split('=')[1]),
                                              reverse=True)
                    for nfe_key, exp_metrics in sorted_optimized:
                        exp_nfe = int(nfe_key.split('=')[1])
                        row_exp = [dataset_name, sigma, 'Optimized', exp_nfe,
                                   f"{exp_metrics['psnr']:.4f}", f"{exp_metrics['ssim']:.4f}",
                                   f"{exp_metrics['lpips']:.4f}"]
                        writer.writerow(row_exp)

        print("Successfully saved results to CSV.")

    def plot_all_results(self):
        print("\nGenerating result plots...")
        for dataset_name, sigma_data in self.all_results.items():
            for sigma, data in sigma_data.items():
                for metric in ['psnr', 'ssim', 'lpips']:

                    labels, scores = [], []
                    orig_nfe = data['Original Paper']['NFE']
                    labels.append(f'Original\n(NFE={orig_nfe})')
                    scores.append(data['Original Paper']['metrics'][metric])

                    # Sort optimized results by NFE
                    sorted_optimized = sorted(data['Optimized'].items(), key=lambda item: int(item[0].split('=')[1]),
                                              reverse=True)

                    for name, exp_data in sorted_optimized:
                        labels.append(f'Optimized\n({name})')
                        scores.append(exp_data[metric])

                    x = np.arange(len(labels))
                    fig, ax = plt.subplots(figsize=(14, 7))
                    bar_colors = ['#B22222'] + ['#4682B4'] * len(data['Optimized'])
                    bars = ax.bar(x, scores, color=bar_colors)

                    # LPIPS is the only metric where lower is better
                    is_lower_better = metric == 'lpips'
                    direction = '(Lower is Better)' if is_lower_better else '(Higher is Better)'
                    unit = ' (dB)' if metric == 'psnr' else ''
                    ax.set_ylabel(f'Average {metric.upper()}{unit}', fontsize=12)
                    ax.set_title(f'{dataset_name} (Sigma={sigma}): {metric.upper()} Comparison\n{direction}',
                                 fontsize=14)
                    ax.set_xticks(x, labels, fontsize=9)
                    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)

                    # Adjust Y-axis limits for better visualization
                    if scores and not np.isnan(scores).any():
                        min_val, max_val = min(scores), max(scores)
                        padding = max((max_val - min_val) * 0.15, 0.001 if metric == 'ssim' else 0.1)
                        ax.set_ylim(min_val - padding, max_val + padding)

                    plt.tight_layout()
                    save_filename = os.path.join(self.save_path, f'{dataset_name}_sigma{sigma}_{metric}_comparison.png')
                    plt.savefig(save_filename)
                    plt.close(fig)
                    print(f"Plot saved to {save_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DMID StepOptim Gaussian Denoising Experiment')
    parser.add_argument('--save_path', default='./results_gaussian_final/', type=str,
                        help='Path to save all results and plots')
    parser.add_argument('--data_root', default='./data', type=str, help='Root directory of datasets')
    args = parser.parse_args()

    config = {
        'diffusion': {'beta_schedule': 'linear', 'beta_start': 0.0001, 'beta_end': 0.02,
                      'num_diffusion_timestamps': 1000, },
        'save': {
            # Path to the pre-trained model checkpoint
            'ddpm_checkpoint': './pre-trained/256x256_diffusion_uncond.pt',
            'photo_path': args.save_path
        },
        'device_id': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    runner = ExperimentRunner(config, save_path=args.save_path)
    runner.run_all_experiments()
    runner.plot_all_results()
    runner.save_results_to_csv("summary_results.csv")