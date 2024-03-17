#!/usr/bin/env python
# coding: utf-8

# PredNet - 5 days adhead forecasting

import os
import numpy as np
from datetime import datetime, timedelta
import json
import sys
sys.path.append("../utils")

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
time_interval_start = config["time_interval_start"]

# start and end datatime for 5day-ahead forecasting
start_date = config["start_test"]
end_date = config["end_test"]

# dataset paths:
path_eac4 = config["path_eac4"]    # EAC4 reanalysis
path_emis = config["path_emis"]    # emission dataset
path_cams = config["path_cams"]    # CAMS forecasts dataset
path_norm = config["path_norm_pred"]    # normalization parameters
path_out_5days = config["path_out_5days"]

trained_prednet = config["trained_prednet"]

features = config["features_prednet"]   # load dictionary: input features - corresponding indices
emis_list = config["emis_list"]         # list for emissions, used as inputs in PredNet
labels = config["labels_prednet"]       # load dictionary: output lables - corresponding indices

# total number of input features in PredNet, including t2m, u10, v10, tcwv, z, bc, oc, aod550
nf = len(features)+len(emis_list)+len(labels)-1

# normalization
# parameters for normalized (save in a dictionary)
with open(path_norm, 'r') as json_file:
    dic_norm = json.load(json_file)

# ======================= load the trained PredNet ============================
# Construct the input layer with no definite frame size.
input_shape = (None,nx,ny,nf)
model = PredNet(input_shape)

def custom_loss(y_actual,y_pred):
    mse = tf.reduce_mean(tf.square(y_pred-y_actual))
    return mse*1e7

# Register custom loss function
custom_objects = {'custom_loss': custom_loss}
model.summary()

# load the trained model
model = keras.models.load_model(trained_prednet, custom_objects=custom_objects)

# =================== Prediction & evaluation ===========================

start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M')

# create a list for all datatimes where 5day-ahead forecasting started
start_list = []
current_date = start_date
while current_date <= end_date:
    start_list.append(current_date)
    current_date += timedelta(hours=time_interval_start)


outname = "pred-{}.npy"

i = 0
# loop over all start time (every 12 hours)
for start in start_list: 
    print("----------- {} -----------".format(start.strftime('%Y%m%d%H')))
    
    # 5-day-ahead forecasting with time interval of 3 hours
    datetime_list = []
    current_date = start
    for i_ahead in range(41):
        datetime_list.append(current_date)
        current_date += timedelta(hours=time_interval)
    
    batch_size = 1
    Nframe = 1
    testGen = sample_generator_prednet(path_eac4, path_emis, datetime_list, nx, ny, Nframe, batch_size, features, emis_list, labels, dic_norm)
    
    # read cams prediciton
    filename = path_cams.format(start.strftime('%Y%m%d%H'))
    print(filename)
    cams_init = np.load(filename)
    cams_pm2p5_new = cams_init[...,0]         # PM2.5 forecasts from CAMS
    cams_aod550_new = cams_init[...,1]        # AOD550 forecasts from CAMS
    
    # save the initial variable
    outpath = path_out_5days.format(start.strftime('%Y%m%d%H'))
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # load the PM2.5 and AOD550 forecasts from  CAMS
    filename = outpath+outname.format((start).strftime('%Y%m%d%H'))
    init = np.array([cams_pm2p5_new, cams_aod550_new])
    init = init.transpose((1, 2, 0))
    np.save(filename, init)
    print('saved to', filename)
    
    cams_pm2p5_new = (cams_pm2p5_new-dic_norm['pm2p5'][0])/(dic_norm['pm2p5'][1]-dic_norm['pm2p5'][0])
    cams_aod550_new = (cams_aod550_new-dic_norm['aod550'][0])/(dic_norm['aod550'][1]-dic_norm['aod550'][0])
    
    # loop over 5 days (ahead forecasting)
    for ahead in range(1, 41):
        
        data_test = testGen.__getitem__(ahead)
        testX = data_test[0]
        testY = np.squeeze(data_test[1])
        if ahead == 1:
            # use the same initial condition with CAMS
            testX[0,-1,::,::,-1] = cams_aod550_new # input aod550 only
        if ahead > 1:
            # iterative forecasting
            testX[0,-1,::,::,-1] = predY[...,-1]
            
        predY = np.squeeze(model.predict(testX, verbose=0))
        
        # scale back
        predY1 = predY.copy()
        predY1[...,0] = predY1[...,0]*(dic_norm['pm2p5'][1]-dic_norm['pm2p5'][0])+dic_norm['pm2p5'][0]
        predY1[...,1] = predY1[...,1]*(dic_norm['aod550'][1]-dic_norm['aod550'][0])+dic_norm['aod550'][0]
            
        filename = outpath+outname.format((start+timedelta(hours=ahead*time_interval)).strftime('%Y%m%d%H'))
        print('saved to', filename)
        np.save(filename, predY1)
        
    i = i+1