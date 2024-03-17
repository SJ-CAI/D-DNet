#!/usr/bin/env python
# coding: utf-8

# D-DNet - operaitonal forecasting

import os
import numpy as np
import time
import json
from datetime import datetime, timedelta

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
from model_define import PredNet, DANet
from sample_generators import sample_generator_prednet, sample_generator_danet

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# ====================================== load related parameters ==========================

# Load the configuration file
config = load_config('../config.yaml')

# load forecast interval
time_interval = config["time_interval"]

# start and end datatime for operational forecasting
start_date = config["start_test"]
end_date = config["end_test"]

da_freq = config["da_freq"]

# path for results
inpath_eac4_loc = config["path_eac4_loc"]            # longitude and latitude of EAC4 dataset
inpath_refer = config["path_eac4"]    # EAC4 reanalysis as reference
inpath_emis = config["path_emis"]    # emission dataset
inpath_obs = config["path_modis"]                # modis dataset, AOD550 datallite observations

path_norm_pred = config["path_norm_pred"]                   # normalize parameters for PredNet
path_norm_da = config["path_norm_da"]               # normalize parameters for MODIS dataset

path_out_oper = config["path_out_oper"]

# path for trained model
trained_prednet = config["trained_prednet"]
trained_danet = config["trained_danet"]

# read time, lon and lat data from nc file
loc = np.load(inpath_eac4_loc)
lon_mesh = loc['lon']
lat_mesh = loc['lat']
nx, ny = lon_mesh.shape


# ========================== setup fpr PredNet ==========================
# for fewtures and labels
features_pred = config["features_prednet"]
labels_pred = config["labels_prednet"]
emis_list = config["emis_list"]
nf_pred = len(features_pred)+len(emis_list)+len(labels_pred)-1   # not using PM2.5 for model input

# parameters for normalised (prednet)
with open(path_norm_pred, 'r') as json_file:
    dic_norm_pred = json.load(json_file)

# Construct the input layer with no definite frame size.
input_shape = shape=(None, nx,ny,nf_pred)
model_pred = PredNet(input_shape)

def custom_loss(y_actual,y_pred):
    mse = tf.reduce_mean(tf.square(y_pred-y_actual))
    return mse*1e7
# Register custom loss function
custom_objects = {'custom_loss': custom_loss}

model_pred.summary()

# load the trained model
model_pred = keras.models.load_model(trained_prednet, custom_objects=custom_objects)

# ========================== setup fpr DANet ==========================

features_da = config["features_danet"]
nf_da = len(features_da) + 1  # for satellite data

# load normalisation parameters
with open(path_norm_da, 'r') as json_file:
    dic_norm_da = json.load(json_file)

# Construct the input layer with no definite frame size.
input_shape_da = (None, nx,ny,nf_da)
model_da = DANet(input_shape_da, 1)   # 1. only assimilate aod
model_da.summary()

# load the trained DA model
model_da = keras.models.load_model(trained_danet, custom_objects=custom_objects)

# ========================== operational forecasting using D-DNet ==========================

# use data in 2011-2017 for training and validation
start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M')

times_index_test = []
current_date = start_date
while current_date <= end_date:
    times_index_test.append(current_date)
    current_date += timedelta(hours=time_interval)

# prediction generator
batch_size = 1
Nframe = 1
predGen = sample_generator_prednet(inpath_refer, inpath_emis, times_index_test, nx, ny, Nframe,
                                                       batch_size, features_pred, emis_list, labels_pred, dic_norm_pred)

# inpath to save operational forecasting results
path_out_oper = path_out_oper.format(times_index_test[0].strftime('%Y%m%d%H'))
if not os.path.exists(path_out_oper):
    os.makedirs(path_out_oper)

path_out_oper = path_out_oper + "oper-{}.npy"

file_list_test = []
for date_time in times_index_test:
    file_list_test.append(path_out_oper.format(date_time.strftime('%Y%m%d%H')))

daGen = sample_generator_danet(path_out_oper, inpath_obs, inpath_refer,
                                                             file_list_test, batch_size, nx, ny, dic_norm_da,
                                                             n_feature=nf_da)

# Record the starting time
start_time = time.time()

# --------------- Operational Forecasting --------------------------
nt = predGen.__len__()
print(nt, nx, ny)

# one-step ahead prediction
operY = np.empty((nt, nx, ny, 2))  # operational forecasting results
testY = np.empty((nt, nx, ny, 2))  # truth for reference

# inpath_cams = "/DATA/CAMS/cams_resample/start_{}/cams-{}.npy"
inpath_cams = config["path_cams"]
start = times_index_test[0]
print(start)
# load initial condition (CAMS)
filename = inpath_cams.format(start.strftime('%Y%m%d%H'))
operY[0, ...] = np.load(filename)
filename = path_out_oper.format(times_index_test[0].strftime('%Y%m%d%H'))
np.save(filename, operY[0,...]) # scale back
print('saved to', filename)

# Nornmalization
operY[0, ..., 0] = (operY[0, ..., 0] - dic_norm_pred['pm2p5'][0]) / (
            dic_norm_pred['pm2p5'][1] - dic_norm_pred['pm2p5'][0])
operY[0, ..., 1] = (operY[0, ..., 1] - dic_norm_pred['aod550'][0]) / (
            dic_norm_pred['aod550'][1] - dic_norm_pred['aod550'][0])

for i in range(1, nt):
    if i % 24 == 0:
        print("----------- {} -----------".format(times_index_test[i]))

    data_test = predGen.__getitem__(i)
    testX = data_test[0]
    testY[i, ...] = np.squeeze(data_test[1])

    # iterative forecasting
    if i > 0:
        testX[0, -1, ::, ::, -1] = operY[i - 1, ..., 1]  # only AOD for the iterative forecasting

    operY[i, ...] = np.squeeze(model_pred.predict(testX, verbose=0))  # output both PM2.5 and AOD

    # save operational forecasting results
    filename = path_out_oper.format(times_index_test[i].strftime('%Y%m%d%H'))
    operY_unscale = operY[i, ...].copy()
    operY_unscale[..., 0] = operY_unscale[..., 0] * (dic_norm_pred['pm2p5'][1] - dic_norm_pred['pm2p5'][0]) + \
                            dic_norm_pred['pm2p5'][0]
    operY_unscale[..., 1] = operY_unscale[..., 1] * (dic_norm_pred['aod550'][1] - dic_norm_pred['aod550'][0]) + \
                            dic_norm_pred['aod550'][0]
    np.save(filename, operY_unscale)  # scale back
    print('saved to', filename)

    # data assimilation
    if (i % da_freq == 0) and (i > 0):
        # prediction has been saved and load from the generator function
        data_test_da = daGen.__getitem__(i)
        testX = data_test_da[0]  # AOD pred and obs differences

        # testY[i,...,1] = np.squeeze(data_test_da[1])
        daY = np.squeeze(model_da.predict(testX, verbose=0))
        # unscale from da
        daY_unscale = daY.copy()
        daY_unscale = daY_unscale * dic_norm_da['aod550_diff'][1]  # assimilate aod only

        daY_update = testX[0, 0, ..., 0]
        # unscale
        daY_update = daY_update * dic_norm_da['aod550'][1]
        # update
        daY_update = daY_update + daY_unscale
        operY_unscale[..., 1] = daY_update  # update aod550 only
        np.save(filename, operY_unscale)  # scale back
        print('DA: saved to', filename)

        # update the prediction  # aod550 only
        operY[i, ..., 1] = (daY_update - dic_norm_pred['aod550'][0]) / (
                    dic_norm_pred['aod550'][1] - dic_norm_pred['aod550'][0])

# Record the ending time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")