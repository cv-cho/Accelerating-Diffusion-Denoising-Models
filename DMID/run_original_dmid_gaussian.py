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
import csv
import pandas as pd

from utils.utils_image import tensor2uint, calculate_psnr, calculate_ssim
from utils.utils_sampling import generalized_steps
from step_optim import NoiseScheduleVP, StepOptim
from guided_diffusion.unet import UNetModel
from Data import Dataset


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear": return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    raise NotImplementedError(beta_schedule)


def data_transform(X): return 2 * X - 1.0


def data_transform_reverse(X): return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DenoisingEngine(object):

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device_id'])
        self.model = UNetModel(image_size=256, in_channels=3, model_channels=256, out_channels=6, num_res_blocks=2,
                               attention_resolutions=(32, 16, 8), dropout=0.0, channel_mult=(1, 1, 2, 2, 4, 4),
                               num_classes=None, use_checkpoint=False, num_heads=4, num_head_channels=64)
        self.model.to(self.device)
        # Load the pre-trained model checkpoint
        self.model.load_state_dict(torch.load(config['save']['ddpm_checkpoint'], map_location=self.device))
        self.model.eval()
        if torch.cuda.device_count() > 1: self.model = nn.DataParallel(self.model)
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        betas = get_beta_schedule(beta_schedule=config['diffusion']['beta_schedule'],
                                  beta_start=config['diffusion']['beta_start'],
                                  beta_end=config['diffusion']['beta_end'],
                                  num_diffusion_timesteps=config['diffusion']['num_diffusion_timesteps'])
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.alphas_cumprod = (1.0 - self.betas).cumprod(dim=0)
        print('____________________________ Denoising Engine Initialized ____________________________')

    def sample_image(self, x_noisy, T_start, custom_seq, eta=0.85):
        t = torch.full((x_noisy.shape[0],), T_start - 1, device=self.device, dtype=torch.long)
        alpha_t_cumprod = self.alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)
        x_N = x_noisy * alpha_t_cumprod.sqrt()
        xs = generalized_steps(x=x_N, seq=custom_seq, model=self.model, b=self.betas, eta=eta)
        return xs[0][-1]

    def run_single_evaluation(self, dataloader, T_start, nfe):
        results = {'psnr': [], 'ssim': [], 'lpips': []}

        # Use linear spacing for the "Original" schedule
        sequence = np.linspace(start=0, stop=T_start - 1, num=nfe + 1).astype(int)

        with torch.no_grad():
            for index, (noisy_tensor, clean_tensor) in enumerate(
                    tqdm(dataloader, desc=f"Original Schedule, NFE={nfe}")):
                noisy_img, clean_img = noisy_tensor.to(self.device), clean_tensor.to(self.device)
                h, w = noisy_img.shape[2], noisy_img.shape[3]
                factor, padh, padw = 64, (64 - h % 64) % 64, (64 - w % 64) % 64
                noise_img_padded = F.pad(noisy_img, (0, padw, 0, padh), 'reflect')
                x_noisy_transformed = data_transform(noise_img_padded)

                denoised_tensor = self.sample_image(x_noisy_transformed, T_start, sequence, eta=0.85)
                # Clamp values to avoid numerical issues
                denoised_tensor = torch.nan_to_num(denoised_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                denoised_img = data_transform_reverse(denoised_tensor)

                restored_np = tensor2uint(denoised_img[:, :, :h, :w])
                clean_np = tensor2uint(clean_img)
                results['psnr'].append(calculate_psnr(restored_np, clean_np, border=0))
                results['ssim'].append(calculate_ssim(restored_np, clean_np, border=0))

                # Ensure tensors are on the correct device for LPIPS
                results['lpips'].append(self.lpips_fn(data_transform(denoised_img[:, :, :h, :w].to(self.device)),
                                                      data_transform(clean_img)).item())

        return {metric: np.mean(values) for metric, values in results.items()}


class ExperimentRerunner:

    def __init__(self, config, save_path):
        self.engine = DenoisingEngine(config)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        # T_start (N) lookup table
        self.n_list = [33, 57, 115, 215, 291, 348, 393]
        self.sigma_list = [15, 25, 50, 100, 150, 200, 250]

    def get_t_start(self, sigma):
        closest_sigma = min(self.sigma_list, key=lambda x: abs(x - sigma))
        idx = self.sigma_list.index(closest_sigma)
        return self.n_list[idx]

    def get_experiments_from_csv(self, csv_filepath):
        df = pd.read_csv(csv_filepath)
        # Rerun "Original" schedule only for NFE values present in "Optimized" runs
        df_optimized = df[df['Method'] == 'Optimized']

        experiments = {}
        for (dataset, sigma), group in df_optimized.groupby(['Dataset', 'Sigma']):
            if dataset not in experiments:
                experiments[dataset] = {}
            nfe_list = sorted(group['NFE'].unique(), reverse=True)
            experiments[dataset][sigma] = nfe_list

        return experiments

    def run(self, experiments_to_run):
        all_new_results = []
        header = ['Dataset', 'Sigma', 'Method', 'NFE', 'PSNR', 'SSIM', 'LPIPS']
        all_new_results.append(header)

        for name, sigma_data in experiments_to_run.items():
            for sigma, nfe_list in sigma_data.items():
                print(f"\n{'=' * 40}\nRerunning Original for: {name} (Sigma={sigma})\n{'=' * 40}")

                # Data path is hardcoded relative to the script
                data_path = f'./data/{name}'
                if not os.path.exists(data_path):
                    print(f"Warning: Dataset path not found: {data_path}. Skipping.")
                    continue

                dataset = Dataset(path_clean=data_path, path_noise=data_path, noise_sigma=sigma, opt=name)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
                T_start = self.get_t_start(sigma)

                for nfe in nfe_list:
                    metrics = self.engine.run_single_evaluation(dataloader, T_start, nfe - 1)
                    row = [name, sigma, 'Original', nfe,  # Save the original NFE from the CSV
                           f"{metrics['psnr']:.4f}", f"{metrics['ssim']:.4f}", f"{metrics['lpips']:.4f}"]
                    all_new_results.append(row)

        return all_new_results

    def save_to_csv(self, data, filename="summary_results_original_rerun.csv"):
        save_filepath = os.path.join(self.save_path, filename)
        print(f"\nSaving new original schedule results to {save_filepath}...")
        with open(save_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        print("Successfully saved results to CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DMID Original Schedule Runner')
    # Path to the CSV file containing the 'Optimized' results
    parser.add_argument('--input_csv', default='./results_gaussian_final/summary_results.csv', type=str,
                        help='Input CSV file with optimized results')
    parser.add_argument('--save_path', default='./results_gaussian_rerun/', type=str, help='Path to save new results')
    args = parser.parse_args()

    config = {
        'diffusion': {'beta_schedule': 'linear', 'beta_start': 0.0001, 'beta_end': 0.02,
                      'num_diffusion_timesteps': 1000},
        'save': {
            # Path to the pre-trained model checkpoint
            'ddpm_checkpoint': './pre-trained/256x256_diffusion_uncond.pt'
        },
        'device_id': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    try:
        rerunner = ExperimentRerunner(config, save_path=args.save_path)
        experiments = rerunner.get_experiments_from_csv(args.input_csv)
        final_data = rerunner.run(experiments)
        rerunner.save_to_csv(final_data)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_csv}' not found. Please make sure it is in the correct directory.")
    except Exception as e:
        print(f"An error occurred: {e}")