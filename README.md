# D-DNet - operational PM2.5 forecasting
This is the official repository for the D-DNet paper. 

This repository include training, forecasting, evaluation, and valization codes for PredNet, DANet, and D-DNet.

# Installation
The downloaded files shall be organized as the following hierarchy:
```plain
├── root
│   ├── environment.yml
│   ├── config.yaml
│   ├── corr_analysis
│   ├── PredNet
│   │   ├── PredNet_training.py
│   │   ├── PredNet_forecasting_long.py
│   │   ├── PredNet_forecasting_5days.py
│   ├── DANet
│   │   ├── DANet_training.py
│   │   ├── DANet_testing.py
│   ├── model
│   │   ├── PredNet_best.h5
│   │   ├── DANet_best.h5
│   ├── D-DNet
│   │   ├── DDNet_operal_forecasting.py
│   ├── results
│   ├── visual
```
Please use this commond line to install important libary dependencies.
```
conda env create -f environment.yml
```
# The trained model
Pre-trained models (PredNet and DANet) are provided in this repository at ```model/PredNet_best.h5``` and ```model/DANet_best.h5```
- The PredNet is trained through the ```PredNet/PredNet_training.py``` using three datesets: EAC4, meterological, emission (~1 TB). The full training process takes roughly 24 hours (depends on different GPU). For more detials about training datasets, see Data requirements below.
- The DANet is another neural network trained based on the PredNet forecasts and satellite observations (MOD08 and MYD08). The code of training DANet is ```DANet/DANet_training.py```. It takes around 3 hours to train this model (depends on different GPU)

# Data requirements
There several datasets required for this project, including:
- EAC4 reanalysis dataset (reference): The EAC4 (ECMWF Atmospheric Composition Reanalysis) dataset is employed as the reference or “ground truth” of atmospheric composition states in our study. (https://www.ecmwf.int/en/forecasts/dataset/cams-global-reanalysis)
- Emission:
  - CAMS-GLOB-ANT dataset (emissions): The CAMS-GLOB-ANT dataset, developed as part of the Copernicus Atmosphere Monitoring Service (CAMS), offers monthly global emissions data for 36 compounds. (https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-emission-inventories)
  - CAMS-TEMPO dataset (emission temporal profile maps):The Copernicus Atmosphere Monitoring Service TEMPOral profiles (CAMS-TEMPO) dataset provides detailed temporal profiles for global and European emissions in atmospheric chemistry modelling. (https://eccad.sedoo.fr/)
- CAMS global atmospheric composition forecasts dataset (baseline): CAMS provides global forecasts for more than 50 chemical species and 7 different types of aerosols, using numerical atmospheric models and data assimilation. It offers global 5-day forecasts twice daily at 00 and 12 UTC. (https://www.ecmwf.int/en/forecasts/dataset/cams-global-atmospheric-composition-forecasts)
- MOD08 and MYD08 datasets (satellite observations): MOD08 and MYD08, both part of NASA's MODIS (Moderate Resolution Imaging Spectroradiometer) data collection, provide daily information about atmospheric properties. (https://www.earthdata.nasa.gov/)
  
# Prepare data paths
edit the paths in the config.yaml file to point to your dataset paths:
```python
{
  path_eac4: '../ddnet_demo/dataset/EAC4/*'        # EAC4 reanalysis dataset
  path_emis: "../ddnet_demo/dataset/emission/*"    # emission dataset
  path_cams: "../ddnet_demo/dataset/CAMS/*"        # CAMS dataset
  path_modis: "../ddnet_demo/dataset/NASA/*"       # satellite dataset
}
```
# License
The D-DNet is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14987021.svg)](https://doi.org/10.5281/zenodo.14987021)

