import argparse
import os
import time
from torch import nn
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import csv

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
        self.model = UNetModel()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(config['save']['ddpm_checkpoint'], map_location=self.device))
        self.model.eval()
        if torch.cuda.device_count() > 1: self.model = nn.DataParallel(self.model)

        betas = get_beta_schedule(beta_schedule=config['diffusion']['beta_schedule'],
                                  beta_start=config['diffusion']['beta_start'],
                                  beta_end=config['diffusion']['beta_end'],
                                  num_diffusion_timesteps=config['diffusion']['num_diffusion_timesteps'])
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.alphas_cumprod = (1.0 - self.betas).cumprod(dim=0)

        # Initialize noise schedule and step optimizer
        self.ns = NoiseScheduleVP(schedule='discrete', alphas_cumprod=self.alphas_cumprod)
        self.step_optim = StepOptim(self.ns)
        self.sequence_cache = {}
        print('____________________________ Denoising Time Measurement Engine Initialized ____________________________')

    def sample_image(self, x_noisy, T_start, custom_seq, eta=0.85):
        t = torch.full((x_noisy.shape[0],), T_start - 1, device=self.device, dtype=torch.long)
        alpha_t_cumprod = self.alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)
        x_N = x_noisy * alpha_t_cumprod.sqrt()

        xs = generalized_steps(x=x_N, seq=custom_seq, model=self.model, b=self.betas, eta=eta)
        # Return the final denoised image
        return xs[0][-1]

    def measure_inference_time(self, dataloader, T_start, nfe, schedule_type='original'):
        cache_key = (T_start, nfe, schedule_type)
        if cache_key in self.sequence_cache:
            sequence = self.sequence_cache[cache_key]
        else:
            if schedule_type == 'original':
                # Generate original linear/uniform step sequence
                skip = T_start // nfe if nfe > 0 else 1
                seq = list(range(0, T_start, skip))
                sequence = np.array(seq[:nfe] + [T_start - 1])
                sequence = np.unique(sequence)
            else:  # optimized
                # Generate new optimized sequence using StepOptim
                print(f"    Generating new optimized sequence for T_start={T_start}, NFE={nfe}...")
                self.ns.T = T_start / 1000.0
                self.step_optim.T = T_start / 1000.0
                t_steps, _ = self.step_optim.get_ts_lambdas(N=nfe, eps=1 / 1000.0, initType='unif_t')
                sequence = np.unique(np.flip((t_steps.numpy() * 1000).astype(int)))
            self.sequence_cache[cache_key] = sequence

        actual_nfe = len(sequence) - 1 if len(sequence) > 0 else 0

        # --- GPU Warm-up ---
        print("    Warming up GPU...")
        with torch.no_grad():
            try:
                dummy_tensor, _ = next(iter(dataloader))
                dummy_img = dummy_tensor.to(self.device)
                h, w = dummy_img.shape[2], dummy_img.shape[3]
                padh, padw = (64 - h % 64) % 64, (64 - w % 64) % 64
                dummy_img_padded = F.pad(dummy_img, (0, padw, 0, padh), 'reflect')
                x_dummy_transformed = data_transform(dummy_img_padded)

                # Run dummy samples through the model to warm up CUDA kernels
                for _ in range(2):
                    _ = self.sample_image(x_dummy_transformed, T_start, sequence, eta=0.85)
            except Exception as e:
                # Fallback to a random tensor if dataloader is empty or fails
                print(f"    Warm-up failed, using random tensor: {e}")
                dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
                x_dummy_transformed = data_transform(dummy_input)
                for _ in range(2):
                    _ = self.sample_image(x_dummy_transformed, T_start, sequence, eta=0.85)

        torch.cuda.synchronize()

        # --- Timed Run ---
        print("    Starting timed run...")

        total_time = 0.0
        num_images = 0

        with torch.no_grad():
            # Use CUDA events for accurate timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            for index, (noisy_tensor, _) in enumerate(
                    tqdm(dataloader, desc=f"T={T_start}, NFE={nfe}, type={schedule_type}")):
                noisy_img = noisy_tensor.to(self.device)
                h, w = noisy_img.shape[2], noisy_img.shape[3]
                # Pad image dimensions to be divisible by 64
                padh, padw = (64 - h % 64) % 64, (64 - w % 64) % 64
                noise_img_padded = F.pad(noisy_img, (0, padw, 0, padh), 'reflect')

                x_noisy_transformed = data_transform(noise_img_padded)

                denoised_avg = self.sample_image(x_noisy_transformed, T_start, sequence, eta=0.85).to(self.device)

            end_event.record()
            # Wait for all CUDA kernels to finish before reading the time
            torch.cuda.synchronize()

            total_time = start_event.elapsed_time(end_event) / 1000.0
            num_images = len(dataloader)

        avg_time = total_time / num_images if num_images > 0 else 0

        print(f"    Finished. Total time: {total_time:.4f}s, Avg per image: {avg_time:.4f}s")
        return total_time, avg_time, actual_nfe


class ExperimentRunner:

    def __init__(self, config, save_path):
        self.engine = DenoisingEngine(config)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        # T_start (N) lookup table based on noise sigma
        self.n_list = [33, 57, 115, 215, 291, 348, 393]
        self.sigma_list = [15, 25, 50, 100, 150, 200, 250]

        # NFE lookup table
        # Format: { sigma: (Original NFE, Optimal NFE, Degradation NFE) }
        self.experiment_nfe_table = {
            'CBSD68': {
                50: (28, 22, 18), 100: (43, 39, 37), 150: (72, 55, 51),
                200: (69, 43, 39), 250: (98, 54, 47)
            },
            'Kodak': {
                50: (57, 34, 25), 100: (107, 53, 41), 150: (143, 82, 76),
                200: (173, 51, 42), 250: (196, 54, 47)
            },
            'McMaster': {
                50: (57, 41, 35), 100: (107, 77, 54), 150: (143, 110, 84),
                200: (173, 112, 86), 250: (196, 128, 98)
            },
            'Imagenet': {
                50: (57, 41, 17), 100: (107, 106, 7), 150: (145, 84, 58),
                200: (173, 112, 86), 250: (196, 128, 98)
            }
        }

    # Find the closest sigma in the list to get the corresponding T_start (N)
    def get_t_start(self, sigma):
        closest_sigma = min(self.sigma_list, key=lambda x: abs(x - sigma))
        idx = self.sigma_list.index(closest_sigma)
        return self.n_list[idx]

    def run_time_measurement_experiment(self):
        datasets_to_run = ['CBSD68', 'Kodak', 'McMaster', 'Imagenet']
        sigmas_to_run = [50, 100, 150, 200, 250]

        csv_path = os.path.join(self.save_path, "inference_time_results.csv")
        print(f"Starting time measurement. Results will be saved to {csv_path}")

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write CSV header
            writer.writerow(
                ['Dataset', 'Sigma', 'ScheduleType', 'NFE_Target', 'NFE_Actual', 'TotalTime_s', 'AvgTimePerImage_s'])

            for name in datasets_to_run:
                if name not in self.experiment_nfe_table:
                    print(f"Skipping dataset {name}, not found in NFE table.")
                    continue

                for sigma in sigmas_to_run:
                    if sigma not in self.experiment_nfe_table[name]:
                        print(f"Skipping sigma {sigma} for {name}, not found in NFE table.")
                        continue

                    print(f"\n{'=' * 40}\nMeasuring: {name} (Sigma={sigma})\n{'=' * 40}")

                    data_path = f'./data/{name}'
                    if not os.path.exists(data_path):
                        print(f"Warning: Dataset path not found: {data_path}. Skipping dataset.")
                        break  # Stop processing this dataset if path not found

                    try:
                        dataset = Dataset(path_clean=data_path, path_noise=data_path, noise_sigma=sigma, opt=name)
                        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
                        if len(dataloader) == 0:
                            print(f"Warning: Dataloader for {name} is empty. Skipping.")
                            continue
                    except Exception as e:
                        print(f"Error loading dataset {name}: {e}. Skipping.")
                        continue

                    T_start = self.get_t_start(sigma)
                    # Get the NFE values for this specific experiment
                    nfe_orig, nfe_opt, nfe_deg = self.experiment_nfe_table[name][sigma]

                    # Define the three scenarios to test
                    experiments_to_run = [
                        ('original', nfe_orig, 'Original'),
                        ('optimized', nfe_opt, 'Optimal'),
                        ('optimized', nfe_deg, 'Degradation')
                    ]

                    for (schedule_type, nfe, schedule_name) in experiments_to_run:
                        print(f"  -> Running {schedule_name} (Type: {schedule_type}, Target NFE: {nfe})")

                        total_time, avg_time, actual_nfe = self.engine.measure_inference_time(
                            dataloader, T_start, nfe, schedule_type
                        )

                        # Write the timing result to the CSV
                        writer.writerow(
                            [name, sigma, schedule_name, nfe, actual_nfe, f"{total_time:.4f}", f"{avg_time:.4f}"])

        print("\nAll time measurements finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DMID StepOptim Gaussian Denoising TIME MEASUREMENT')
    parser.add_argument('--save_path', default='./results_time_measurement/', type=str,
                        help='Path to save time measurement CSV')
    args = parser.parse_args()

    # Main configuration dictionary
    config = {
        'diffusion': {'beta_schedule': 'linear', 'beta_start': 0.0001, 'beta_end': 0.02,
                      'num_diffusion_timesteps': 1000, },
        'save': {
            # Path to the pre-trained model checkpoint
            'ddpm_checkpoint': './pre-trained/256x256_diffusion_uncond.pt',
            'photo_path': args.save_path
        },
        'device_id': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    if config['device_id'] == 'cpu':
        print("WARNING: Running on CPU. Time measurements will not be representative of GPU performance.")

    runner = ExperimentRunner(config, save_path=args.save_path)

    runner.run_time_measurement_experiment()