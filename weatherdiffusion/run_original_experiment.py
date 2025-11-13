import argparse
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import lpips
import PIL
import csv

# --- Dependency Imports ---
import models
import datasets
import utils
import utils.utils_image
from step_optim import NoiseScheduleVP, StepOptim


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class DenoisingEngine:
    # Utility class to perform denoising and return evaluation metrics.

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use the specific initialization from the original Weather Diffusion
        self.diffusion = models.DenoisingDiffusion(args, config)
        self.model_wrapper = models.DiffusiveRestoration(self.diffusion, args, config)

        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        self.ns = NoiseScheduleVP(schedule='discrete', betas=self.diffusion.betas)
        self.step_optim = StepOptim(self.ns)
        self.sequence_cache = {}
        print(f"Engine initialized for {args.resume}")

    def run_single_evaluation(self, dataloader, nfe, schedule_type='original', num_samples_to_run=-1):
        T_start = self.config.diffusion.num_diffusion_timesteps
        cache_key = (T_start, nfe, schedule_type)
        if cache_key in self.sequence_cache:
            sequence = self.sequence_cache[cache_key]
        else:
            if schedule_type == 'original':
                # Replicate the original paper's uniform step division
                sequence = range(0, T_start, T_start // nfe)
            else:  # optimized
                print(f"Generating optimized sequence for NFE={nfe}...")
                self.step_optim.T = 1.0
                t_steps, _ = self.step_optim.get_ts_lambdas(N=nfe, eps=1 / T_start, initType='unif_t')
                sequence = np.unique(np.flip((t_steps.numpy() * (T_start - 1)).astype(int)))
            self.sequence_cache[cache_key] = sequence

        # `range` length is the NFE, `np.unique` length-1 is the NFE
        actual_nfe = len(sequence) if schedule_type == 'original' else len(sequence) - 1

        psnr_list, ssim_list, lpips_list = [], [], []
        with torch.no_grad():
            for i, (data_cat, img_ids) in enumerate(tqdm(dataloader, desc=f"NFE={actual_nfe}, type={schedule_type}")):
                if num_samples_to_run > 0 and i >= num_samples_to_run: break

                x = data_cat[:, :3, :, :].to(self.device)
                y = data_cat[:, 3:, :, :].to(self.device)

                # Replicate the original overlapping diffusive restoration logic
                p_size = self.config.data.image_size
                h_list, w_list = self.model_wrapper.overlapping_grid_indices(x, output_size=p_size, r=self.args.grid_r)
                corners = [(hi, wi) for hi in h_list for wi in w_list]
                x_init = torch.randn(x.size(), device=self.device)
                x_output = utils.sampling.generalized_steps_overlapping(x_init, x, sequence, self.diffusion.model,
                                                                        self.diffusion.betas, eta=0., corners=corners,
                                                                        p_size=p_size)[0][-1]
                x_output_restored = utils.sampling.inverse_data_transform(x_output)

                # Convert to NumPy for PSNR/SSIM
                restored_np = utils.utils_image.tensor2uint(x_output_restored)
                gt_np = utils.utils_image.tensor2uint(y)
                psnr_list.append(utils.utils_image.calculate_psnr(restored_np, gt_np))
                ssim_list.append(utils.utils_image.calculate_ssim(gt_np, restored_np))

                # Convert to GPU Tensor for LPIPS
                lpips_input_restored = utils.data_transform(x_output_restored).to(self.device)
                lpips_input_gt = utils.data_transform(y)
                lpips_list.append(self.lpips_fn(lpips_input_restored, lpips_input_gt).item())

        return {
            'psnr': np.mean(psnr_list), 'ssim': np.mean(ssim_list), 'lpips': np.mean(lpips_list)
        }, actual_nfe


class ExperimentRunner:
    # Orchestrates the experiment and manages results.

    def __init__(self, save_path):
        self.all_results = {}
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def run_all_experiments(self):
        model_configs = [
            {'name': 'WeatherDiff64', 'config': 'allweather.yml', 'resume': 'WeatherDiff64.pth.tar'},
            {'name': 'WeatherDiff128', 'config': 'allweather128.yml', 'resume': 'WeatherDiff128.pth.tar'}
        ]
        datasets_to_run = ['snow', 'raindrop', 'rainfog']
        num_samples = 50  # Number of samples to run per experiment for speed

        for m_config in model_configs:
            model_name = m_config['name']
            self.all_results[model_name] = {}

            args = argparse.Namespace(
                **{'config': m_config['config'], 'resume': m_config['resume'], 'grid_r': 16, 'seed': 61})
            with open(os.path.join("configs", args.config), "r") as f:
                config = yaml.safe_load(f)
            config = dict2namespace(config)
            config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            engine = DenoisingEngine(args, config)

            if model_name == 'WeatherDiff64':
                nfe_list_to_run = [10, 8, 6, 4, 2]
            else:  # WeatherDiff128
                nfe_list_to_run = [50, 40, 30, 20, 10]

            for dataset_name in datasets_to_run:
                print(f"\n{'=' * 40}\nStarting Experiment for: {model_name} on {dataset_name}\n{'=' * 40}")
                self.all_results[model_name][dataset_name] = {'Original': {}}

                DATASET = datasets.__dict__[config.data.dataset](config)
                _, dataloader = DATASET.get_loaders(parse_patches=False, validation=dataset_name)

                # This script *only* runs the 'original' schedule
                for nfe in nfe_list_to_run:
                    print(f"--> Running Control Group: NFE={nfe}")
                    orig_metrics, actual_nfe = engine.run_single_evaluation(dataloader, nfe, 'original',
                                                                            num_samples_to_run=num_samples)
                    self.all_results[model_name][dataset_name]['Original'][f'NFE={actual_nfe}'] = orig_metrics

                self.plot_single_result(model_name, dataset_name)
        print("\nAll experiments finished.")

    def plot_single_result(self, model_name, dataset_name):
        print(f"\nGenerating plot for {model_name} on {dataset_name}...")
        data = self.all_results[model_name][dataset_name]
        for metric in ['psnr', 'ssim', 'lpips']:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot a line graph showing the performance trend of the Original schedule
            orig_data = sorted(data['Original'].items(), key=lambda item: int(item[0].split('=')[1]))
            x_orig = [int(item[0].split('=')[1]) for item in orig_data]
            y_orig = [item[1][metric] for item in orig_data]
            ax.plot(x_orig, y_orig, marker='o', linestyle='--', label='Original Schedule')

            is_lower_better = metric == 'lpips'
            direction = '(Lower is Better)' if is_lower_better else '(Higher is Better)'
            unit = ' (dB)' if metric == 'psnr' else ''

            ax.set_xlabel('Number of Function Evaluations (NFE)', fontsize=12)
            ax.set_ylabel(f'Average {metric.upper()}{unit}', fontsize=12)
            ax.set_title(f'{model_name} on {dataset_name}: {metric.upper()} vs. NFE\n{direction}', fontsize=14)
            ax.grid(True, which='both', linestyle=':')
            ax.legend(fontsize=12)

            plt.tight_layout()
            save_filename = os.path.join(self.save_path, f'{model_name}_{dataset_name}_{metric}_original_schedule.png')
            plt.savefig(save_filename)
            plt.close(fig)
            print(f"Plot saved to {save_filename}")

    def save_results_to_csv(self, filename="summary_weather_results_original.csv"):
        save_filepath = os.path.join(self.save_path, filename)
        print(f"\nSaving all results to {save_filepath}...")
        header = ['Model', 'Dataset', 'Method', 'NFE', 'PSNR', 'SSIM', 'LPIPS']

        with open(save_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for model_name, model_data in self.all_results.items():
                for dataset_name, dataset_data in model_data.items():
                    # Save only the 'Original' schedule results
                    sorted_original = sorted(dataset_data['Original'].items(),
                                             key=lambda item: int(item[0].split('=')[1]), reverse=True)
                    for nfe_key, orig_metrics in sorted_original:
                        nfe = int(nfe_key.split('=')[1])
                        row = [model_name, dataset_name, 'Original', nfe,
                               f"{orig_metrics['psnr']:.4f}", f"{orig_metrics['ssim']:.4f}",
                               f"{orig_metrics['lpips']:.4f}"]
                        writer.writerow(row)
        print("Successfully saved results to CSV.")


if __name__ == "__main__":
    # This script evaluates the performance of the *original* schedule
    # across a range of NFEs.
    parser = argparse.ArgumentParser(description='Weather Diffusion Original Schedule Evaluation')
    parser.add_argument('--save_path', default='./results_weather_original_only/', type=str,
                        help='Path to save all results and plots')
    args = parser.parse_args()

    runner = ExperimentRunner(save_path=args.save_path)
    runner.run_all_experiments()
    runner.save_results_to_csv("summary_weather_results_original.csv")