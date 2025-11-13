# Accelerating Diffusion Denoising Models (Official Code)

This repository contains the official code and replication instructions for the paper "Accelerating Diffusion-based Denoising Model with Optimized Time Steps". We provide the core `step_optim.py` algorithm and the experiment scripts required to reproduce our results for the DDRM, DMID, and WeatherDiffusion models.

## 1. Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/cv-cho/Accelerating-Diffusion-Denoising-Models.git](https://github.com/cv-cho/Accelerating-Diffusion-Denoising-Models.git)
    cd Accelerating-Diffusion-Denoising-Models
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The different experiments (DDRM, DMID, WeatherDiffusion) were originally run in separate environments. We provide a combined `requirements.txt` for convenience. If you encounter dependency conflicts, we recommend creating separate conda environments as specified in the respective original repositories.*

---

## 2. Replicating DDRM (Gaussian Denoising) Experiments

This section provides instructions to replicate the Gaussian Denoising experiments (Tables 1-4, Figure 3) from our paper, which are based on the Denoising Diffusion Restoration Models (DDRM) [1].

### 2.1. Prerequisites

1.  **Pre-trained Model:**
    * This experiment requires the unconditional ImageNet 256x256 model from OpenAI.
    * Download the model: [`256x256_diffusion_uncond.pt`](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)
    * Create the required directory structure inside the `ddrm/` folder and place the model there:
      ```
      Accelerating-Diffusion-Denoising-Models/ddrm/exp/logs/imagenet/
      └── 256x256_diffusion_uncond.pt
      ```
    * The `exp` folder should be at the same level as your `run_ddrm_experiment.py` script.

2.  **Datasets:**
    * Download the `CBSD68`, `Kodak`, `McMaster`, and `Imagenet` (validation) test sets.
    * Place each dataset inside the `ddrm/data/` folder (e.g., `Accelerating-Diffusion-Denoising-Models/ddrm/data/CBSD68/`).

3.  **Required Code Dependencies:**
    * This experiment relies on the core framework from the [official DDRM repository](https://github.com/bahjat-kawar/ddrm).
    * Please copy the following folders from the official DDRM repo into your `Accelerating-Diffusion-Denoising-Models/ddrm/` directory:
        * `guided_diffusion/`
        * `runners/`
        * `functions/`
        * `datasets/`
        * `configs/`
    * (Alternatively, use the files already included in this repository if available.)

### 2.2. Replicating Paper Results

The DDRM Gaussian denoising results (PSNR, SSIM, LPIPS) in our paper (Tables 1-4, Figure 3) can be replicated by running `run_ddrm_experiment.py`. This script evaluates both the 'Original' (uniform) schedule and our 'Optimized' (StepOptim) schedule.

```bash
# Navigate to the ddrm/ directory (Required)
cd ddrm

# Run the main StepOptim experiment
python run_ddrm_experiment.py --save_path ./results_ddrm_final/
```

* This script iterates through $\sigma = [15, 25, 50]$ and NFE levels of `[20, 16, 12, 8, 4]`.
* All quantitative results are saved to `results_ddrm_final/summary_ddrm_results.csv`.
* All comparative line plots are saved as `*.png` files in the same directory.

### 2.3. Inference Speed Measurement

The inference speed (wall-clock time) comparison in our paper (Table 6) can be replicated using `inference_time_ddrm.py`.

```bash
# Make sure you are in the ddrm/ directory
cd ddrm

# Run the time measurement script
python inference_time_ddrm.py --save_path ./results_ddrm_time/
```

* Results are saved to `results_ddrm_time/inference_time_ddrm_results.csv`.

---

## 3. Replicating DMID (Gaussian Denoising) Experiments

This section guides how to replicate the Gaussian noise removal experiments (Table 5, Figure 4) using the DMID [2] approach.

*(Note: The original DMID paper uses the same unconditional `guided_diffusion` model as DDRM. This experiment uses the same model.)*

### 3.1. Prerequisites

1.  **Pre-trained Model:**
    * Download the same [OpenAI pre-trained model](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) as used for DDRM.
    * Place the downloaded `256x256_diffusion_uncond.pt` file in the `Accelerating-Diffusion-Denoising-Models/DMID/pre-trained/` folder.

2.  **Datasets:**
    * Prepare the `CBSD68`, `Kodak`, `McMaster`, and `Imagenet` (Validation) datasets.
    * Place each dataset inside the `Accelerating-Diffusion-Denoising-Models/DMID/data/` folder (e.g., `.../DMID/data/CBSD68/`).

3.  **Required Code Dependencies:**
    * This experiment requires `Data.py`, `utils/utils_sampling.py`, and `guided_diffusion`.
    * Please copy `guided_diffusion` from the [DDRM repo](https://github.com/bahjat-kawar/ddrm) and `Data.py`/`utils/` from the [DMID repo](https://github.com/Li-Tong-621/DMID) into the `Accelerating-Diffusion-Denoising-Models/DMID/` folder.
    * (Alternatively, use the files already included in this repository if available.)

### 3.2. Replicating Paper Results

#### A. Generating Optimized Schedule (StepOptim) Results

The `run_stepoptim_dmid_gaussian.py` script runs both the `Original` (Baseline) and `Optimized` (StepOptim) schedules, generating `summary_results.csv` to reproduce Table 5 and Figure 4.

```bash
# Navigate to the DMID/ directory (Required)
cd DMID

# Run the StepOptim experiment (Baseline + Optimized)
python run_stepoptim_dmid_gaussian.py --save_path ./results_gaussian_final/
```
* This script runs experiments for $\sigma = [50, 100, 150, 200, 250]$ noise levels.
* The `Original` schedule uses the `T_start` and `S_t` (NFE) defined by the DMID authors [2].
* Results are saved to `results_gaussian_final/summary_results.csv` and as `*.png` graphs.

#### B. Re-running the Original Schedule (Optional)

The `run_original_dmid_gaussian.py` script re-runs the 'Original' schedule for the *exact same NFEs* as the 'Optimized' schedule for a fair comparison.

```bash
# Make sure you are in the DMID/ directory
cd DMID

# Rerun the Original schedule
python run_original_dmid_gaussian.py --input_csv ./results_gaussian_final/summary_results.csv --save_path ./results_gaussian_rerun/
```

* Results are saved to `results_gaussian_rerun/summary_results_original_rerun.csv`.

### 3.3. Inference Speed Measurement

The inference speed (wall-clock time) comparison in our paper (Table 6) can be replicated using `inference_time.py`.

```bash
# Make sure you are in the DMID/ directory
cd DMID

# Run time measurement
python inference_time.py --save_path ./results_time_measurement/
```

* This script measures time for the NFE values (Original, Optimal, Degradation) from our paper.
* Results are saved to `results_time_measurement/inference_time_results.csv`.

## 4. Replicating WeatherDiffusion (Adverse Weather) Experiments

This section guides how to replicate the adverse weather removal experiments (Tables 7-8, Figure 5) based on the WeatherDiffusion [3] framework.

### 4.1. Prerequisites

1.  **Pre-trained Models:**
    * Download the official pre-trained models from the original authors' links:
        * [`WeatherDiff64.pth.tar`](https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff64.pth.tar)
        * [`WeatherDiff128.pth.tar`](https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff128.pth.tar)
    * Place the downloaded `.pth.tar` files directly inside the `Accelerating-Diffusion-Denoising-Models/weatherdiffusion/` folder.

2.  **Datasets:**
    * This experiment uses the `Snow100K`, `RainDrop`, and `Outdoor-Rain` datasets.
    * The original framework expects a specific directory structure. Create the following structure inside the `weatherdiffusion/` folder:
      ```
      Accelerating-Diffusion-Denoising-Models/weatherdiffusion/
      └── scratch/
          └── ozan/
              └── data/
                  ├── snow100k/
                  │   ├── snowtest100k_L.txt
                  │   ├── (gt/ ...)
                  │   └── (input/ ...)
                  ├── raindrop/
                  │   ├── test/
                  │   │   ├── raindroptesta.txt
                  │   │   ├── (gt/ ...)
                  │   │   └── (input/ ...)
                  │   └── train/
                  └── outdoor-rain/
                      ├── test1.txt
                      ├── (gt/ ...)
                      └── (input/ ...)
      ```
    * Our scripts are configured to find data at this path (`config.data.data_dir = './scratch/ozan/data'`).

3.  **Required Code Dependencies:**
    * This experiment relies on the core framework from the [official WeatherDiffusion repository](https://github.com/IGITUGraz/WeatherDiffusion).
    * Please copy the following folders from the official repo into your `Accelerating-Diffusion-Denoising-Models/weatherdiffusion/` directory:
        * `models/`
        * `datasets/`
        * `utils/`
        * `configs/` (containing `allweather.yml` and `allweather128.yml`)
    * (Alternatively, use the files already included in this repository if available.)
  
### 4.2. Replicating Paper Results

The WeatherDiffusion results (PSNR, SSIM, LPIPS) in our paper (Tables 7-8, Figure 5) can be replicated by running `run_stepoptim_experiment.py`.

```bash
# Navigate to the weatherdiffusion/ directory (Required)
cd weatherdiffusion

# Run the StepOptim experiment (Baseline + Optimized)
python run_stepoptim_experiment.py --save_path ./results_weather_final/
```

* This script will evaluate on `snow`, `raindrop`, and `rainfog` test sets (50 samples each).
* **WeatherDiff64** is evaluated with NFEs of `[10, 8, 6, 4, 2]`.
* **WeatherDiff128** is evaluated with NFEs of `[50, 40, 30, 20, 10]`.
* All quantitative results are saved to `results_weather_final/summary_weather_results.csv`.
* All comparative bar graphs are saved as `*.png` files in the same directory.

### 4.3. Inference Speed Measurement

The inference speed (wall-clock time) comparison in our paper (Table 6) can be replicated using `inference_time_weather.py`.

```bash
# Make sure you are in the weatherdiffusion/ directory
cd weatherdiffusion

# Run time measurement
python inference_time_weather.py --save_path ./results_weather_time/
```

* This script runs both 'Original' and 'Optimized' schedules for all NFE levels specified in the paper.
* Results are saved to `results_time_weather/inference_time_weather_results.csv`.

---
### References

[1] B. Kawar, M. Elad, S. Ermon, and J. Song. "Denoising diffusion restoration models." *Advances in neural information processing systems* 35 (2022): 23593-23606.

[2] T. Li, H. Feng, L. Wang, et al. "Stimulating diffusion model for image denoising via adaptive embedding and ensembling." *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2024).

[3] O. Özdenizci, and R. Legenstein. "Restoring vision in adverse weather conditions with patch-based denoising diffusion models." *IEEE transactions on pattern analysis and machine intelligence* 45.8 (2023): 10346-10357.
