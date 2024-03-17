#!/usr/bin/env python
# coding: utf-8

# PredNet - long forecasting

import os
import numpy as np
from datetime import datetime, timedelta
import json
import sys
sys.path.append("../utils")

# for training and validation
import tensorflow as tf
from tensorflow import keras

import utils, sample_generators, model_define

from importlib import reload
reload(utils)
reload(sample_generators)
reload(model_define)

from utils import load_config
from sample_generators import sample_generator_prednet
from model_define import PredNet

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# ====================================== load related parameters ==========================

# Load the configuration file
config = load_config('../config.yaml')

nx = config["nx"]
ny = config["ny"]

# load forecast interval
time_interval = config["time_interval"]

# start and end datatime for 5day-ahead forecasting
start_date = config["start_test"]
end_date = config["end_test"]

# dataset paths:
path_eac4 = config["path_eac4"]  # EAC4 reanalysis
path_emis = config["path_emis"]  # emission dataset
path_cams = config["path_cams"]  # CAMS forecasts dataset
path_norm = config["path_norm_pred"]  # normalization parameters
path_out_long = config["path_out_long"]

trained_prednet = config["trained_prednet"]

# define a dictionary for features: corresponding indices
features = config["features_prednet"]
# list for emissions, used as inputs in PredNet
emis_list = config["emis_list"]
# define a dictionary for labels: corresponding indices
labels = config["labels_prednet"]

# total number of input features in PredNet, including t2m, u10, v10, tcwv, z, bc, oc, aod550
nf = len(features) + len(emis_list) + len(labels) - 1

# normalization
# parameters for normalized (save in a dictionary)
with open(path_norm, 'r') as json_file:
    dic_norm = json.load(json_file)

# ======================= load the trained PredNet ============================
# Construct the input layer with no definite frame size.
input_shape = (None, nx, ny, nf)
model = PredNet(input_shape)

def custom_loss(y_actual, y_pred):
    mse = tf.reduce_mean(tf.square(y_pred - y_actual))
    return mse * 1e7


# Register custom loss function
custom_objects = {'custom_loss': custom_loss}
model.summary()

# load the trained model
model = keras.models.load_model(trained_prednet, custom_objects=custom_objects)


# =================== Prediction & evaluation ===========================

start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M')

# interatively conducting long-term forecasting from start_date to end_date
datetime_list = []
current_date = start_date
while current_date <= end_date:
    datetime_list.append(current_date)
    current_date += timedelta(hours=time_interval)

batch_size = 1
Nframe = 1
testGen = sample_generator_prednet(path_eac4, path_emis, datetime_list, nx, ny, Nframe, batch_size, features, emis_list, labels, dic_norm)
ns = testGen.__len__()

# read cams prediciton
filename = path_cams.format(start_date.strftime('%Y%m%d%H'))
print(filename)
cams_init = np.load(filename)
cams_pm2p5_new = cams_init[..., 0]  # PM2.5 forecasts from CAMS
cams_aod550_new = cams_init[..., 1]  # AOD550 forecasts from CAMS

# save the initial variable
outpath = path_out_long.format(start_date.strftime('%Y%m%d%H'))
if not os.path.exists(outpath):
    os.makedirs(outpath)
    
filename = outpath+'pred-{}.npy'.format((start_date).strftime('%Y%m%d%H'))
init = np.array([cams_pm2p5_new, cams_aod550_new])
init = init.transpose((1, 2, 0))
np.save(filename, init)
print('saved to', filename)

cams_pm2p5_new = (cams_pm2p5_new-dic_norm['pm2p5'][0])/(dic_norm['pm2p5'][1]-dic_norm['pm2p5'][0])
cams_aod550_new = (cams_aod550_new-dic_norm['aod550'][0])/(dic_norm['aod550'][1]-dic_norm['aod550'][0])

# long-term forecasting
for ahead in range(1, ns):
    
    data_test = testGen.__getitem__(ahead)
    testX = data_test[0]
    testY = np.squeeze(data_test[1])
    if ahead == 1:
        # same initial condition as CAMS
        testX[0,-1,::,::,-1] = cams_aod550_new # input aod550 only
    if ahead > 1:
        # iterative forecasting
        testX[0,-1,::,::,-1] = predY[...,-1]
        
    predY = np.squeeze(model.predict(testX, verbose=0))

    # scale back
    predY1 = predY.copy()
    predY1[...,0] = predY1[...,0]*(dic_norm['pm2p5'][1]-dic_norm['pm2p5'][0])+dic_norm['pm2p5'][0]
    predY1[...,1] = predY1[...,1]*(dic_norm['aod550'][1]-dic_norm['aod550'][0])+dic_norm['aod550'][0]
    
    # save x hour ahead iterative forecasting results
    filename = outpath+'pred-{}.npy'.format((start_date+timedelta(hours=ahead*time_interval)).strftime('%Y%m%d%H'))
    print('saved to', filename)
    np.save(filename, predY1)