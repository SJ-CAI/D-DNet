#!/usr/bin/env python
# coding: utf-8

# DANet - testing

import os
import numpy as np
import json
from datetime import datetime, timedelta
import random

import tensorflow as tf
from tensorflow import keras

import sys
sys.path.append("../utils")

from importlib import reload
import utils, sample_generators, model_define

reload(utils)
reload(sample_generators)
reload(model_define)

from utils import load_config
from sample_generators import sample_generator_danet
from model_define import DANet

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# ====================================== load related parameters ==========================

# Load the configuration file
config = load_config('../config.yaml')

# load forecast interval
time_interval = config["time_interval"]
time_interval_start = config["time_interval_start"]

# start and end datatime for 5day-ahead forecasting
start_date = config["start_test"]
end_date = config["end_test"]

# path for files
inpath_eac4_loc = config["path_eac4_loc"]            # longitude and latitude of EAC4 dataset
inpath_refer = config["path_eac4"]                     # eac4 reanalysis dataset as reference
inpath_pred = config["path_out_5days"]+"pred-{}.npy"   # 5day-aheaf forecasts from PredNet
inpath_obs = config["path_modis"]                # modis dataset, AOD550 datallite observations

path_norm = config["path_norm_da"]                  # normalize parameters for MODIS dataset

outpath_da = config["path_out_da"]                     # temporal folder for DANet testing

# trained DANet
trained_danet = config["trained_danet"]

features = config["features_danet"]
n_ch = len(features) + 1  # for satellite-AOD

# load lon and lat information
loc = np.load(inpath_eac4_loc)
lon_mesh = loc['lon']
lat_mesh = loc['lat']
nx, ny = lon_mesh.shape

# parameters for normalized (save in a dictionary)
with open(path_norm, 'r') as json_file:
    dic_norm = json.load(json_file)
print(dic_norm)

# ======================= load the trained PredNet ============================

input_shape = (None, nx,ny,n_ch)
num_out = 1
model_da = DANet(input_shape, 1)

def custom_loss(y_actual,y_pred):
    mse = tf.reduce_mean(tf.square(y_pred-y_actual))
    return mse*1e7
# Register custom loss function
custom_objects = {'custom_loss': custom_loss}

model_da = keras.models.load_model(trained_danet, custom_objects=custom_objects)
model_da.summary()

# =================== DANet tesing ===========================

start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M') - timedelta(days=5)

start_list = []
current_date = start_date
while current_date <= end_date:
    start_list.append(current_date)
    current_date += timedelta(hours=time_interval_start)

# randomly selecting 100 samples for DANet testing
start_list = random.sample(start_list, 100)
n_start = len(start_list)

n_ahead = 41
for start_date in start_list:
    outpath_da = config["path_out_da"]  # reset temporal folder for DANet testing
    outpath_da = outpath_da.format(start_date.strftime('%Y%m%d%H'))
    if not os.path.exists(outpath_da):
        os.makedirs(outpath_da)

    for ahead in range(n_ahead):
        current_date = start_date + timedelta(hours=time_interval*ahead)

        filename = inpath_pred.format(start_date.strftime('%Y%m%d%H'), current_date.strftime('%Y%m%d%H'))
        file_list_test = [filename]

        batch_size = 1
        testGen = sample_generator_danet(inpath_pred, inpath_obs, inpath_refer,
                                                                       file_list_test, batch_size, nx, ny, dic_norm,
                                                                       n_feature=n_ch)
        predY = np.empty((nx, ny))  # before da
        daY = np.empty((nx, ny))  # after da

        data_test = testGen.__getitem__(0)

        testX = data_test[0]
        daY = np.squeeze(model_da.predict(testX, verbose=0))
        predY = np.squeeze(testX[0, 0, ..., 0])  # errorneous prediction

        predY_update = predY.copy()

        # update
        max_o = dic_norm['aod550'][1]
        max_d = dic_norm['aod550_diff'][1]
        predY_update = predY_update * max_o + daY * max_d

        filename = outpath_da + "da-{}.npy".format(current_date.strftime('%Y%m%d%H'))
        print(filename)
        np.save(filename, predY_update)