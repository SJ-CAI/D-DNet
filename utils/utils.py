#!/usr/bin/env python
# coding: utf-8

# utils

import numpy as np
from importlib import reload
import forecasting_metrics
reload(forecasting_metrics)
from forecasting_metrics import evaluate

# ====================== load configuration ============================
def load_config(file_path):
    import yaml
    with open(file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

'''
evaluating model forecasts against references in nt instances 
'''
def evaluation_nt(date_list, inpath_pred, inpath_refer, index):

    nt = len(date_list)
    RMSE = []
    R = []

    for i in range(nt):
        it = date_list[i]

        filename = inpath_refer.format(it.strftime('%Y%m%d%H'))
        referY = np.load(filename)[..., index]

        filename = inpath_pred.format(it.strftime('%Y%m%d%H'))
        predY = np.load(filename)[..., index]

        metrics = evaluate(referY, predY, metrics=('rmse', 'cc'))
        RMSE.append(metrics['rmse'])
        R.append(metrics['cc'])

    return RMSE, R

