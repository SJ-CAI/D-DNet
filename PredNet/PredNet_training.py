#!/usr/bin/env python
# coding: utf-8

# PredNet training

import numpy as np
import time
from datetime import datetime, timedelta
import json


import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import sys
sys.path.append("../utils")

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

Nframe = config["nframe"]                    # frame ahead as input data
batch_size = config["batch_size_prednet"]    # batch_size for tarining
epochs = config["epochs_prednet"]            # epochs for training

# load forecast interval
time_interval = config["time_interval"]
time_interval_start = config["time_interval_start"]

# start and end datatime for PredNet training
start_date = config["start_train_prednet"]
end_date = config["end_train_prednet"]

# dataset paths:
path_eac4 = config["path_eac4"]    # EAC4 reanalysis
path_emis = config["path_emis"]    # emission dataset

path_norm = config["path_norm_pred"]    # normalization parameters

# training save path
path_model_save = config["prednet_training_save"]         # save the newly trained PredNet
path_his_training = config["his_prednet_training_save"]   # save training history for PredNet

features = config["features_prednet"]   # load dictionary: input features - corresponding indices
emis_list = config["emis_list"]         # list for emissions, used as inputs in PredNet
labels = config["labels_prednet"]       # load dictionary: output lables - corresponding indices

# total number of input features in PredNet, including t2m, u10, v10, tcwv, z, bc, oc, aod550
nf = len(features)+len(emis_list)+len(labels)-1

# parameters for normalisation (save in a dictionary)
with open(path_norm, 'r') as json_file:
    dic_norm = json.load(json_file)

# =================== prepare training dataset ================================
start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M')

datetime_list = []
current_date = start_date
while current_date <= end_date:
    datetime_list.append(current_date)
    current_date += timedelta(hours=time_interval)

[times_train, times_val] = train_test_split(datetime_list, test_size = 0.1, random_state = 8)
print('length all. training. validation.:', len(datetime_list), len(times_train), len(times_val))

# go check the generator function
trainGen = sample_generator_prednet(path_eac4, path_emis, times_train, nx, ny, Nframe, batch_size, features, emis_list, labels, dic_norm)
valGen = sample_generator_prednet(path_eac4, path_emis, times_val, nx, ny, Nframe, batch_size, features, emis_list, labels, dic_norm)

# check the generator
dataset = trainGen.__getitem__(0)
trainX = dataset[0]
trainY = dataset[1]
print(trainX.shape, trainY.shape)

# ======================= define PredNet for training ============================
# Construct the input layer with no definite frame size.
input_shape = (None,nx,ny,nf)
model = PredNet(input_shape)

def custom_loss(y_actual,y_pred):
    mse = tf.reduce_mean(tf.square(y_pred-y_actual))
    return mse*1e7

# Register custom loss function
custom_objects = {'custom_loss': custom_loss}
model.compile(
    loss=custom_loss, optimizer=keras.optimizers.Adam(learning_rate=0.01),
)
model.summary()

# reset model
tf.keras.backend.clear_session()

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint(filepath=path_model_save, monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min', verbose=1)

# ======================== PredNet training =====================================
# --------------- execution time (start) -------------------
t_start = time.time()
t_cpu_start = time.process_time()

print("[INFO] training w/ generator...")
H = model.fit(trainGen,
    steps_per_epoch = len(times_train) // batch_size,
    validation_data = valGen,
    validation_steps = len(times_val) // batch_size,
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