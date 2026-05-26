# D-DNet: A Dual Deep Neural Network Framework for High-Efficiency Operational PM2.5 and AOD550 Forecasting with Data Assimilation

This is the official repository for the D-DNet paper. It includes training, forecasting, evaluation, and validation code for PredNet, DANet, and the full D-DNet operational system.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14987021.svg)](https://doi.org/10.5281/zenodo.14987021)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SJ-CAI/D-DNet.git
cd D-DNet
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate ddnet
```

### 3. Repository structure

```
├── root
│   ├── environment.yml          # conda environment
│   ├── config.yaml              # data paths configuration
│   ├── PredNet
│   │   ├── PredNet_training.py          # train PredNet
│   │   ├── PredNet_forecasting_long.py  # long-term forecasting (full year)
│   │   ├── PredNet_forecasting_5days.py # 5-day forecast evaluation
│   ├── DANet
│   │   ├── DANet_training.py    # train DANet
│   │   ├── DANet_testing.py     # DANet evaluation
│   ├── model
│   │   ├── PredNet_best.h5      # pre-trained PredNet weights
│   │   ├── DANet_best.h5        # pre-trained DANet weights
│   ├── D-DNet
│   │   ├── DDNet_operal_forecasting.py  # full D-DNet operational pipeline
│   ├── results                  # model outputs
│   ├── visual                   # visualisation notebooks
│   │   ├── visual_operational_forecasting.ipynb          # operational forecast time series and global maps
│   │   ├── visual_pred_compare.ipynb                     # PredNet 5-day forecast evaluation (RMSE, R²)
│   │   ├── visual_pred_compare-iniCAMS-iniEAC4-ave_11-17.ipynb  # PredNet comparison with CAMS/EAC4 initialisation
│   │   ├── visual_dann-ave.ipynb                         # DANet analysis quality evaluation
│   │   ├── visual_da.ipynb                               # data assimilation field visualisation
│   │   ├── regional_analysis.ipynb                       # regional performance breakdown
```

---

## Reproducing the Results

Follow these steps to reproduce the main results in the paper.

### Step 1 — Prepare data

Download the required datasets (see [Data Requirements](#data-requirements)) and update the paths in `config.yaml`:

```yaml
path_eac4:  '../ddnet_demo/dataset/EAC4/*'       # EAC4 reanalysis
path_emis:  '../ddnet_demo/dataset/emission/*'   # emission datasets
path_cams:  '../ddnet_demo/dataset/CAMS/*'       # CAMS operational forecasts
path_modis: '../ddnet_demo/dataset/NASA/*'       # MODIS satellite observations
```

### Step 2 — Train PredNet (or use pre-trained weights)

Pre-trained weights are provided at `model/PredNet_best.h5`. To retrain:

```bash
python PredNet/PredNet_training.py
```

Training uses EAC4 reanalysis, meteorological fields, and emission data (~1 TB total). Training time: ~24 hours on a single GPU.

### Step 3 — Train DANet (or use pre-trained weights)

Pre-trained weights are provided at `model/DANet_best.h5`. To retrain:

```bash
python DANet/DANet_training.py
```

DANet is trained using PredNet forecasts and MODIS AOD550 satellite observations (MOD08/MYD08). Training time: ~3 hours on a single GPU.

### Step 4 — Run D-DNet operational forecasting

```bash
python D-DNet/DDNet_operal_forecasting.py
```

This runs the full D-DNet pipeline: PredNet 5-day forecasts initialised every 12 hours, with DANet AOD550 assimilation applied at each cycle.

### Step 5 — Evaluate and visualise

```bash
# 5-day forecast evaluation (PredNet vs CAMS)
python PredNet/PredNet_forecasting_5days.py

# DANet evaluation
python DANet/DANet_testing.py
```

Visualisation notebooks are provided in the `visual/` directory:

| Notebook | Description |
|---|---|
| `visual_operational_forecasting.ipynb` | Operational forecast results — RMSE/R² time series, spatial maps, and latitudinal error profiles for D-DNet vs CAMS 4D-Var (full year 2019) |
| `visual_pred_compare.ipynb` | 5-day forecast evaluation — RMSE/R² vs lead time for PredNet vs CAMS, averaged over 100 random start cases |
| `visual_pred_compare-iniCAMS-iniEAC4-ave_11-17.ipynb` | Comparison of PredNet initialized from CAMS vs EAC4 initial conditions |
| `visual_dann-ave.ipynb` | DANet vs PredNet averaged comparison — RMSE/R² over DANet test cases |
| `visual_da.ipynb` | Data assimilation visualisation — EAC4, PredNet forecast, MODIS satellite, and DANet analysis fields |
| `regional_analysis.ipynb` | Regional performance analysis — RMSE/R² across 20 global regions |

To run a notebook:

```bash
cd visual
jupyter notebook visual_operational_forecasting.ipynb
```

---

## Data Requirements

| Dataset | Description | Source |
|---|---|---|
| EAC4 | ECMWF Atmospheric Composition Reanalysis. Used as training reference and evaluation baseline. 0.75° × 0.75°, sub-daily. | [ECMWF ADS](https://www.ecmwf.int/en/forecasts/dataset/cams-global-reanalysis) |
| CAMS-GLOB-ANT | Monthly global anthropogenic emissions for 36 compounds, 0.1° × 0.1°. Black Carbon and Organic Carbon used for PM2.5 and AOD550 forecasting. | [Copernicus ADS](https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-emission-inventories) |
| CAMS-TEMPO | Temporal emission profiles (monthly/daily/hourly) used to disaggregate CAMS-GLOB-ANT to hourly scale. | [ECCAD](https://eccad.sedoo.fr/) |
| CAMS operational forecasts | Global 5-day atmospheric composition forecasts (00 and 12 UTC). Used as benchmark baseline. | [ECMWF ADS](https://www.ecmwf.int/en/forecasts/dataset/cams-global-atmospheric-composition-forecasts) |
| MOD08 / MYD08 | MODIS daily AOD550 from Terra and Aqua satellites. Used as DA observational input for DANet. | [NASA Earthdata](https://www.earthdata.nasa.gov/) |

---

## Pre-trained Models

Pre-trained models for PredNet and DANet are available in the `model/` directory and via Zenodo (DOI: [10.5281/zenodo.14987021](https://doi.org/10.5281/zenodo.14987021)).

---

## License

D-DNet is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation.
