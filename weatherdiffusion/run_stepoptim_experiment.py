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
from step_optim import NoiseScheduleVP, StepOptim
from utils import utils_image


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict): new_value = dict2namespace(value)
        else: new_value = value
        setattr(namespace, key, new_value)
    return namespace


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
    # Performs denoising operations and returns evaluation metrics.

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model using the original Weather Diffusion setup
        self.diffusion = models.DenoisingDiffusion(args, config)
        self.model = models.DiffusiveRestoration(self.diffusion, args, config)

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
                # Original schedule: uniform steps
                sequence = range(0, T_start, T_start // nfe)
            else:  # optimized
                print(f"Generating optimized sequence for NFE={nfe}...")
                self.step_optim.T = 1.0
                t_steps, _ = self.step_optim.get_ts_lambdas(N=nfe, eps=1 / T_start, initType='unif_t')
                sequence = np.unique(np.flip((t_steps.numpy() * (T_start - 1)).astype(int)))
            self.sequence_cache[cache_key] = sequence

        actual_nfe = len(sequence) if schedule_type == 'original' else len(sequence) - 1

        psnr_list, ssim_list, lpips_list = [], [], []
        with torch.no_grad():
            for i, (data_cat, img_ids) in enumerate(tqdm(dataloader, desc=f"NFE={actual_nfe}, type={schedule_type}")):
                if num_samples_to_run > 0 and i >= num_samples_to_run: break

                x = data_cat[:, :3, :, :].to(self.device)
                y = data_cat[:, 3:, :, :].to(self.device)

                # Run the overlapping patch-based restoration
                p_size = self.config.data.image_size
                h_list, w_list = self.model.overlapping_grid_indices(x, output_size=p_size, r=self.args.grid_r)
                corners = [(hi, wi) for hi in h_list for wi in w_list]
                x_init = torch.randn(x.size(), device=self.device)
                x_output = utils.sampling.generalized_steps_overlapping(x_init, x, sequence, self.diffusion.model,
                                                                        self.diffusion.betas, eta=0., corners=corners,
                                                                        p_size=p_size)[0][-1]
                x_output_restored = utils.sampling.inverse_data_transform(x_output)

                restored_np = utils.utils_image.tensor2uint(x_output_restored)
                gt_np = utils.utils_image.tensor2uint(y)

                psnr_list.append(utils.utils_image.calculate_psnr(restored_np, gt_np))
                ssim_list.append(utils.utils_image.calculate_ssim(gt_np, restored_np))
                lpips_input_restored = utils.data_transform(x_output_restored).to(self.device)
                lpips_input_gt = utils.data_transform(y)  # y is already on device
                lpips_list.append(self.lpips_fn(lpips_input_restored, lpips_input_gt).item())

        return {
            'psnr': np.mean(psnr_list), 'ssim': np.mean(ssim_list), 'lpips': np.mean(lpips_list)
        }, actual_nfe


class ExperimentRunner:
    # Orchestrates the entire experiment, manages results, and saves outputs.

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
        num_samples = 50 # Run on a subset of samples for speed

        for m_config in model_configs:
            model_name = m_config['name']
            self.all_results[model_name] = {}

            args = argparse.Namespace(
                **{'config': m_config['config'], 'resume': m_config['resume'], 'grid_r': 16, 'seed': 61})
            with open(os.path.join("configs", args.config), "r") as f:
                config = yaml.safe_load(f)
            config = dict2namespace(config)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config.device = device
            engine = DenoisingEngine(args, config)

            if model_name == 'WeatherDiff64':
                nfe_control, nfe_exp_list = 10, [10, 8, 6, 4, 2]
            else:  # WeatherDiff128
                nfe_control, nfe_exp_list = 50, [50, 40, 30, 20, 10]

            for dataset_name in datasets_to_run:
                print(f"\n{'=' * 40}\nStarting Experiment for: {model_name} on {dataset_name}\n{'=' * 40}")
                self.all_results[model_name][dataset_name] = {'Original': {}, 'Optimized': {}}

                DATASET = datasets.__dict__[config.data.dataset](config)
                _, dataloader = DATASET.get_loaders(parse_patches=False, validation=dataset_name)

                # Run baseline (Original) schedule once for control
                print(f"--> Running Control Group: NFE={nfe_control}")
                orig_metrics, _ = engine.run_single_evaluation(dataloader, nfe_control, 'original',
                                                               num_samples_to_run=num_samples)
                self.all_results[model_name][dataset_name]['Original'][f'NFE={nfe_control}'] = orig_metrics

                # Run experimental (Optimized) schedule for all NFE values
                for nfe_exp in nfe_exp_list:
                    print(f"--> Running Experimental Group: NFE={nfe_exp}")
                    opt_metrics, actual_nfe = engine.run_single_evaluation(dataloader, nfe_exp, 'optimized',
                                                                           num_samples_to_run=num_samples)
                    self.all_results[model_name][dataset_name]['Optimized'][f'NFE={actual_nfe}'] = opt_metrics

                self.plot_single_result(model_name, dataset_name)

        print("\nAll experiments finished.")

    def save_results_to_csv(self, filename="summary_weather_results.csv"):
        # Save all collected results to a single CSV file.
        save_filepath = os.path.join(self.save_path, filename)
        print(f"\nSaving all results to {save_filepath}...")

        header = ['Model', 'Dataset', 'Method', 'NFE', 'PSNR', 'SSIM', 'LPIPS']

        with open(save_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for model_name, model_data in self.all_results.items():
                for dataset_name, dataset_data in model_data.items():
                    # Write control (Original) result
                    orig_nfe_key = list(dataset_data['Original'].keys())[0]
                    orig_nfe = int(orig_nfe_key.split('=')[1])
                    orig_metrics = dataset_data['Original'][orig_nfe_key]
                    row_orig = [model_name, dataset_name, 'Original', orig_nfe,
                                f"{orig_metrics['psnr']:.4f}", f"{orig_metrics['ssim']:.4f}",
                                f"{orig_metrics['lpips']:.4f}"]
                    writer.writerow(row_orig)

                    # Write experimental (Optimized) results
                    sorted_optimized = sorted(dataset_data['Optimized'].items(),
                                              key=lambda item: int(item[0].split('=')[1]), reverse=True)
                    for nfe_key, exp_metrics in sorted_optimized:
                        exp_nfe = int(nfe_key.split('=')[1])
                        row_exp = [model_name, dataset_name, 'Optimized', exp_nfe,
                                   f"{exp_metrics['psnr']:.4f}", f"{exp_metrics['ssim']:.4f}",
                                   f"{exp_metrics['lpips']:.4f}"]
                        writer.writerow(row_exp)

        print("Successfully saved results to CSV.")

    def plot_single_result(self, model_name, dataset_name):
        print(f"\nGenerating plot for {model_name} on {dataset_name}...")
        data = self.all_results[model_name][dataset_name]

        for metric in ['psnr', 'ssim', 'lpips']:
            labels, scores = [], []

            # Add control (Original) data
            orig_nfe_key = list(data['Original'].keys())[0]
            labels.append(f'Original\n({orig_nfe_key})')
            scores.append(data['Original'][orig_nfe_key][metric])

            # Add experimental (Optimized) data
            sorted_optimized = sorted(data['Optimized'].items(),
                                      key=lambda item: int(item[0].split('=')[1]), reverse=True)
            for name, exp_data in sorted_optimized:
                labels.append(f'Optimized\n({name})')
                scores.append(exp_data[metric])

            # --- Create Bar Graph ---
            x = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(14, 7))
            bar_colors = ['#B22222'] + ['#4682B4'] * len(data['Optimized']) # Red for control, blue for optimized
            bars = ax.bar(x, scores, color=bar_colors, width=0.6)

            is_lower_better = metric == 'lpips'
            direction = '(Lower is Better)' if is_lower_better else '(Higher is Better)'
            unit = ' (dB)' if metric == 'psnr' else ''

            ax.set_ylabel(f'Average {metric.upper()}{unit}', fontsize=12)
            ax.set_title(f'{model_name} on {dataset_name}: {metric.upper()} Comparison\n{direction}',
                         fontsize=14)
            ax.set_xticks(x, labels, fontsize=10)
            ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)

            # Adjust Y-axis limits for better visibility
            all_scores = scores
            if all_scores and not np.isnan(all_scores).any():
                min_val, max_val = min(all_scores), max(all_scores)
                padding = (max_val - min_val) * 0.1
                ax.set_ylim(min_val - padding, max_val + padding if padding > 0 else max_val + 0.1)

            plt.tight_layout()
            save_filename = os.path.join(self.save_path,
                                         f'{model_name}_{dataset_name}_{metric}_bar_comparison.png')
            plt.savefig(save_filename)
            plt.close(fig)
            print(f"Plot saved to {save_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weather Diffusion StepOptim Experiment Runner')
    parser.add_argument('--save_path', default='./results_weather_final/', type=str,
                        help='Path to save all results and plots')
    args = parser.parse_args()

    runner = ExperimentRunner(save_path=args.save_path)
    runner.run_all_experiments()

    runner.save_results_to_csv("summary_weather_results.csv")