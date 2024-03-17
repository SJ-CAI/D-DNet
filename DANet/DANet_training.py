#!/usr/bin/env python
# coding: utf-8

# # DANet training

import numpy as np
import json
import random
import time
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

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

batch_size = config["batch_size_danet"]    # batch_size for tarining
epochs = config["epochs_danet"]            # epochs for training

# load forecast interval
time_interval = config["time_interval"]
time_interval_start = config["time_interval_start"]

# start and end datatime for DANet training
start_date = config["start_train_danet"]
end_date = config["end_train_danet"]

# path for files
inpath_eac4_loc = config["path_eac4_loc"]            # longitude and latitude of EAC4 dataset
inpath_refer = config["path_eac4"]                     # eac4 reanalysis dataset as reference
# inpath_pred = "/DATA/global-emission-2_11-17_iniCAMS_iAOD/pred/start_{}/"+"pred-{}.npy"   # 5day-aheaf forecasts from PredNet
inpath_pred = config["path_out_5days"]+"pred-{}.npy"   # 5day-aheaf forecasts from PredNet
inpath_obs = config["path_modis"]                # modis dataset, AOD550 datallite observations

# for normalization
path_norm = config["path_norm_da"]                  # normalize parameters for MODIS dataset

# training save path
path_model_save = config["danet_training_save"]         # save the newly trained PredNet
path_his_training = config["his_danet_training_save"]   # save training history for PredNet

features = config["features_danet"]
n_ch = len(features) + 1  # for satellite-AOD

# load lon and lat information
loc = np.load(inpath_eac4_loc)
lon_mesh = loc['lon']
lat_mesh = loc['lat']
nx, ny = lon_mesh.shape

# parameters for normalisation (save in a dictionary)
with open(path_norm, 'r') as json_file:
    dic_norm = json.load(json_file)

start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M')

date_list = []
current_date = start_date
while current_date <= end_date:
    date_list.append(current_date)
    current_date += timedelta(hours=time_interval_start)
print(date_list)

# split data to training and validation
# replace the index for training and valiation as the filename
ahead_list = np.arange(0,16)*3  # (2-days)
file_list_train_val = []
for current_date in date_list:
    for ahead in ahead_list:
        ahead_date = current_date + timedelta(hours=int(ahead))
        if ahead_date <= end_date:
            filename = inpath_pred.format(current_date.strftime('%Y%m%d%H'), ahead_date.strftime('%Y%m%d%H'))
            file_list_train_val.append(filename)
# print(file_list_train_val)

print(len(file_list_train_val))
file_list_train_val = random.sample(file_list_train_val, 5840) # random 50%

[times_index_train, times_index_val] = train_test_split(file_list_train_val, test_size = 0.1, random_state = 8)
print(len(times_index_train), len(times_index_val))

print('\n', [itime for itime in times_index_train[:5]])
print('\n', [itime for itime in times_index_val[-5:]])

# training, validation and test split
trainGen = sample_generator_danet(inpath_pred, inpath_obs, inpath_refer, times_index_train, batch_size, nx, ny, dic_norm, n_feature=n_ch)
valGen = sample_generator_danet(inpath_pred, inpath_obs, inpath_refer, times_index_val, batch_size, nx, ny, dic_norm, n_feature=n_ch)

data = trainGen.__getitem__(0)
print(data[0].shape)
for index in range(n_ch):
    print(index, np.nanmin(data[0][...,index]), np.nanmax(data[0][...,index]))

print(data[1].shape)
for index in range(1):
    print(index, np.nanmin(data[1][...,index]), np.nanmax(data[1][...,index]))

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# Open a strategy scope.
with strategy.scope():
    # Construct the input layer with no definite frame size.
    input_shape = (None, nx,ny,n_ch)
    model_da = DANet(input_shape, 1)
    def custom_loss(y_actual,y_pred):
        mse = tf.reduce_mean(tf.square(y_pred-y_actual))
        return mse*1e7
    # Register custom loss function
    custom_objects = {'custom_loss': custom_loss}

    model_da.compile(
        loss=custom_loss, optimizer=keras.optimizers.Adam(learning_rate=0.01),
    )

model_da.summary()

# reset model
tf.keras.backend.clear_session()

# Define the ModelCheckpoint callback
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
checkpoint = ModelCheckpoint(filepath=path_model_save, monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min', verbose=1)

# --------------- execution time (start) -------------------
t_start = time.time()
t_cpu_start = time.process_time()

print("[INFO] training w/ generator...")
H = model_da.fit(trainGen,
    steps_per_epoch = len(times_index_train) // batch_size,
    validation_data = valGen,
    validation_steps = len(times_index_val) // batch_size,
    epochs = epochs,
    verbose = 1,
    callbacks=[early_stopping, reduce_lr, checkpoint])

# --------------- execution time (end) ------------------------
t_execut  = time.time() - t_start
t_cpu_execut = time.process_time()
print('Execution time:',t_execut, 'seconds')
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(t_execut)))
print()
print('CPU Execution time:',t_cpu_execut, 'seconds')

# --------------- save training history ------------------
np.savez(path_his_training, loss = H.history['loss'], val_loss = H.history['val_loss'])
print('training history saved to', path_his_training)