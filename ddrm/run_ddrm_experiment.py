import argparse
import os
import yaml
import torch
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
import lpips
import PIL
import csv
from natsort import natsorted
import glob
from guided_diffusion.script_util import create_model
# --- Dependency Imports ---
from runners.diffusion import Diffusion
from functions.denoising import efficient_generalized_steps
from functions.svd_replacement import Denoising
from datasets import data_transform, inverse_data_transform
from step_optim import NoiseScheduleVP, StepOptim
from utils.utils_image import tensor2uint, calculate_psnr, calculate_ssim


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class GenericImageDataset(torch.utils.data.Dataset):
    # A dataset that loads images from a folder and adds Gaussian noise on-the-fly.

    def __init__(self, root_dir, image_size, sigma):
        self.image_paths = natsorted(glob.glob(os.path.join(root_dir, '*.*')))
        self.sigma = sigma / 255.0
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        clean_pil = PIL.Image.open(img_path).convert("RGB")
        clean_tensor = self.transform(clean_pil)
        # DDRM requires a class label, providing -1 as a dummy.
        return clean_tensor, -1


class DenoisingEngine:
    # Utility class to perform DDRM denoising operations.

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.runner = Diffusion(args, config, self.device)
        config_dict = vars(self.config.model)
        self.model = create_model(**config_dict)
        # Load the pre-trained DDRM checkpoint
        ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.model = torch.nn.DataParallel(self.model)
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        self.ns = NoiseScheduleVP(schedule='discrete', betas=self.runner.betas)
        self.step_optim = StepOptim(self.ns)
        self.sequence_cache = {}
        print(f"Engine initialized for DDRM")

    def run_single_evaluation(self, dataloader, nfe, schedule_type='original', num_samples_to_run=-1):
        T_start = self.config.diffusion.num_diffusion_timesteps
        cache_key = (T_start, nfe, schedule_type)
        if cache_key in self.sequence_cache:
            sequence = self.sequence_cache[cache_key]
        else:
            if schedule_type == 'original':
                # Baseline: uniform linear spacing
                sequence = np.linspace(start=0, stop=T_start - 1, num=nfe).astype(int)
            else:  # optimized
                print(f"Generating optimized sequence for NFE={nfe}...")
                self.step_optim.T = 1.0

                # Pass N-1 to get_ts_lambdas to get N timesteps (N-1 intervals)
                num_intervals = max(1, nfe - 1)
                t_steps, _ = self.step_optim.get_ts_lambdas(N=num_intervals, eps=1 / T_start, initType='unif_t')

                sequence = np.unique(np.flip((t_steps.numpy() * (T_start - 1)).astype(int)))
            self.sequence_cache[cache_key] = sequence

        actual_nfe = len(sequence)

        psnr_list, ssim_list, lpips_list = [], [], []
        with torch.no_grad():
            for i, (x_orig, _) in enumerate(tqdm(dataloader, desc=f"NFE={actual_nfe}, type={schedule_type}")):
                # Option to run on a subset of the dataset for speed
                if num_samples_to_run > 0 and i * dataloader.batch_size >= num_samples_to_run:
                    print(f"\nEvaluation stopped after reaching approximately {num_samples_to_run} samples.")
                    break

                x_orig = x_orig.to(self.device)
                x_orig_transformed = data_transform(self.config, x_orig)

                # Set up the denoising operator (H) and observation (y)
                H_funcs = Denoising(self.config.data.channels, self.config.data.image_size, self.device)
                y_0 = H_funcs.H(x_orig_transformed)
                y_0 += self.args.sigma_0 * torch.randn_like(y_0)

                x_init = torch.randn_like(x_orig_transformed)

                # Run the efficient generalized sampling steps
                x_output, _ = efficient_generalized_steps(x_init, sequence, self.model, self.runner.betas, H_funcs, y_0,
                                                          self.args.sigma_0, etaB=1.0, etaA=0.85, etaC=0.85)

                x_output_final_batch = x_output[-1]
                x_output_restored_batch = inverse_data_transform(self.config, x_output_final_batch)
                x_orig_restored_batch = inverse_data_transform(self.config, x_orig_transformed)

                for j in range(x_output_restored_batch.shape[0]):
                    restored_img = x_output_restored_batch[j:j + 1]
                    gt_img = x_orig_restored_batch[j:j + 1]

                    restored_np = tensor2uint(restored_img).squeeze()
                    gt_np = tensor2uint(gt_img).squeeze()

                    psnr_list.append(calculate_psnr(restored_np, gt_np))
                    ssim_list.append(calculate_ssim(restored_np, gt_np))

                    # LPIPS requires tensors on the correct device
                    lpips_list.append(self.lpips_fn(restored_img.to(self.device), gt_img).item())

        return {'psnr': np.mean(psnr_list), 'ssim': np.mean(ssim_list), 'lpips': np.mean(lpips_list)}, actual_nfe


class ExperimentRunner:
    # Main class to orchestrate experiments, plot, and save results.

    def __init__(self, save_path):
        self.all_results = {}
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def run_all_experiments(self):
        datasets_to_run = ['CBSD68', 'Kodak', 'McMaster', 'Imagenet']
        sigmas_to_run = [15, 25, 50]
        num_samples = 100  # Number of samples to run per experiment

        for dataset_name in datasets_to_run:
            self.all_results[dataset_name] = {}
            for sigma in sigmas_to_run:
                print(f"\n{'=' * 40}\nStarting Experiment for: {dataset_name} (Sigma={sigma})\n{'=' * 40}")
                self.all_results[dataset_name][sigma] = {'Original': {}, 'Optimized': {}}

                # Set up configuration for this specific run
                args = argparse.Namespace(**{'config': 'imagenet_256.yml', 'exp': 'exp', 'seed': 1234, 'deg': 'deno',
                                             'sigma_0': sigma / 255.0})
                with open(os.path.join("configs", args.config), "r") as f:
                    config = yaml.safe_load(f)
                config = dict2namespace(config)
                config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # IMPORTANT: Disable mixed precision (fp16) for stable results
                config.model.use_fp16 = False

                engine = DenoisingEngine(args, config)

                data_path = f'./data/{dataset_name}'
                if not os.path.exists(data_path):
                    print(f"Warning: Dataset path not found: {data_path}. Skipping.")
                    continue
                dataset = GenericImageDataset(root_dir=data_path, image_size=config.data.image_size, sigma=sigma)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.sampling.batch_size, shuffle=False,
                                                         num_workers=0)

                nfe_list = [20, 16, 12, 8, 4]

                for nfe in nfe_list:
                    print(f"\n--> Running NFE = {nfe}")
                    # Original
                    orig_metrics, _ = engine.run_single_evaluation(dataloader, nfe, 'original',
                                                                   num_samples_to_run=num_samples)
                    self.all_results[dataset_name][sigma]['Original'][nfe] = orig_metrics
                    # Optimized
                    opt_metrics, actual_nfe = engine.run_single_evaluation(dataloader, nfe, 'optimized',
                                                                           num_samples_to_run=num_samples)
                    self.all_results[dataset_name][sigma]['Optimized'][actual_nfe] = opt_metrics

                self.plot_single_result(dataset_name, sigma)
        print("\nAll experiments finished.")

    def plot_single_result(self, dataset_name, sigma):
        print(f"\nGenerating plot for {dataset_name} (Sigma={sigma})...")
        data = self.all_results[dataset_name][sigma]
        for metric in ['psnr', 'ssim', 'lpips']:
            fig, ax = plt.subplots(figsize=(10, 6))

            orig_data = sorted(data['Original'].items())
            x_orig = [item[0] for item in orig_data]
            y_orig = [item[1][metric] for item in orig_data]
            ax.plot(x_orig, y_orig, marker='o', linestyle='--', label='Original Schedule')

            opt_data = sorted(data['Optimized'].items())
            x_opt = [item[0] for item in opt_data]
            y_opt = [item[1][metric] for item in opt_data]
            ax.plot(x_opt, y_opt, marker='s', linestyle='-', label='Optimized Schedule (StepOptim)')

            is_lower_better = metric == 'lpips'
            direction = '(Lower is Better)' if is_lower_better else '(Higher is Better)'
            unit = ' (dB)' if metric == 'psnr' else ''

            ax.set_xlabel('Number of Function Evaluations (NFE)', fontsize=12)
            ax.set_ylabel(f'Average {metric.upper()}{unit}', fontsize=12)
            ax.set_title(f'{dataset_name} (Sigma={sigma}): {metric.upper()} vs. NFE\n{direction}', fontsize=14)
            ax.grid(True, which='both', linestyle=':')
            ax.legend(fontsize=12)
            ax.invert_xaxis()  # Plot NFE from high to low

            plt.tight_layout()
            save_filename = os.path.join(self.save_path, f'{dataset_name}_sigma{sigma}_{metric}_line_comparison.png')
            plt.savefig(save_filename)
            plt.close(fig)
            print(f"Plot saved to {save_filename}")

    def save_results_to_csv(self, filename="summary_ddrm_results.csv"):
        save_filepath = os.path.join(self.save_path, filename)
        print(f"\nSaving all results to {save_filepath}...")
        header = ['Dataset', 'Sigma', 'NFE_requested', 'Method', 'NFE_actual', 'PSNR', 'SSIM', 'LPIPS']

        with open(save_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for dataset_name, sigma_data in self.all_results.items():
                for sigma, data in sigma_data.items():
                    sorted_nfes = sorted(data['Original'].keys(), reverse=True)
                    for nfe in sorted_nfes:
                        # Original
                        orig_metrics = data['Original'][nfe]
                        row_orig = [dataset_name, sigma, nfe, 'Original', nfe, f"{orig_metrics['psnr']:.4f}",
                                    f"{orig_metrics['ssim']:.4f}", f"{orig_metrics['lpips']:.4f}"]
                        writer.writerow(row_orig)

                        # Optimized
                        # Match the requested NFE to the closest actual NFE from the optimized run
                        found = False
                        for actual_nfe, opt_metrics in data['Optimized'].items():
                            if abs(actual_nfe - nfe) <= 2:  # Match if actual NFE is close to requested
                                row_opt = [dataset_name, sigma, nfe, 'Optimized', actual_nfe,
                                           f"{opt_metrics['psnr']:.4f}", f"{opt_metrics['ssim']:.4f}",
                                           f"{opt_metrics['lpips']:.4f}"]
                                writer.writerow(row_opt)
                                found = True
                                break
                        if not found:
                            writer.writerow([dataset_name, sigma, nfe, 'Optimized', 'N/A', 'N/A', 'N/A', 'N/A'])

        print("Successfully saved results to CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDRM StepOptim Experiment Runner')
    parser.add_argument('--save_path', default='./results_ddrm_final/', type=str,
                        help='Path to save all results and plots')
    args = parser.parse_args()

    runner = ExperimentRunner(save_path=args.save_path)
    runner.run_all_experiments()
    runner.save_results_to_csv("summary_ddrm_results.csv")