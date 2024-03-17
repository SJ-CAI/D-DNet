#!/usr/bin/env python
# coding: utf-8

# sample generators for PredNet and DANet

import os
import re
import math 
import numpy as np
from datetime import timedelta
from tensorflow import keras


# ========================== function for global dataset with emission and AOD =======================

# function for sample generation
# original dataset
# currently, this function is only suitable for nlag = 1 cases
# with emission as input
# with AOD as output
# input historical AOD550 only, ouput AOD550 and PM2.5 concentration

class sample_generator_prednet(keras.utils.Sequence):

    def __init__(self, path_eac4, path_emis, indices, nx, ny, nfr, batch_size, features, emis_list, labels, dic_norm, mode="train"):

        self.path_eac4 = path_eac4  # input path for files
        self.path_emis = path_emis  # input path for emission files
        self.indices = indices      # %Y%m%d%H list
        self.nx = nx                # grid number
        self.ny = ny
        self.nfr = nfr              # number of previous frames used for prediction
        self.batch_size = batch_size  # batch size
        self.features = features    # dictionary: features - index
        self.emis_list = emis_list  # list for emission
        self.labels = labels        # dictionary: labels - index
        self.dic_norm = dic_norm    # normalisation

    #  computes the number of batches that this generator is supposed to produce

    def __len__(self):
        return int(math.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_idx = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]

        # read dimensional informations
        nf = len(self.features) + len(self.emis_list) + len(self.labels) - 1 # delete pm2.5 input

        dataX = []
        dataY = []

        for idxi in batch_idx:

            dataXi = np.empty((self.nfr, self.nx, self.ny, nf), dtype='float32')
            dataYi = np.empty((1, self.nx, self.ny, len(self.labels)), dtype='float32')

            # organise one sample - using previous nfr frames
            for xfr in range(0, -self.nfr, -1):
                # ---------- current time step ---------------
                current = idxi + timedelta(hours=xfr*3)
                filename = self.path_eac4.format(current.strftime('%Y%m%d%H'))
                data = np.load(filename)

                # ----------- features ------------
                for i_feature in range(len(self.features)):
                    key = list(self.features.keys())[i_feature]
                    index = self.features[key]
                    value = data[..., index]
                    value = (value-self.dic_norm[key][0])/(self.dic_norm[key][1]-self.dic_norm[key][0])
                    dataXi[0, ..., i_feature] = value

                # ----------- emission ------------
                for i_emis in range(len(self.emis_list)):
                    key = self.emis_list[i_emis]
                    filename = self.path_emis.format(key, current.strftime('%Y%m%d%H'))
                    np_file = np.load(filename)
                    ipoll = np_file.files[-1]
                    value = np.load(filename)[ipoll]
                    value = (value-self.dic_norm[key][0])/(self.dic_norm[key][1]-self.dic_norm[key][0])
                    dataXi[0, ..., i_emis+len(self.features)] = value

                # -------- labels ---------
                for i_label in range(len(self.labels)):
                    key = list(self.labels.keys())[i_label]
                    index = self.labels[key]
                    # ------------- previous time step ------------
                    value = data[..., index]
                    value = (value - self.dic_norm[key][0]) / (self.dic_norm[key][1] - self.dic_norm[key][0])
                    dataYi[0, ..., i_label] = value

                # -------- labels (previous hour)---------
                filename = self.path_eac4.format((current - timedelta(hours=3)).strftime('%Y%m%d%H'))
                data = np.load(filename)
                for i_label in range(1, len(self.labels)): # delete pm2.5 input
                    key = list(self.labels.keys())[i_label]
                    # print(key)
                    index = self.labels[key]
                    # ------------- previous time step ------------
                    value = data[..., index]
                    value = (value - self.dic_norm[key][0]) / (self.dic_norm[key][1] - self.dic_norm[key][0])
                    dataXi[0, ..., i_label-1 + len(self.features)+len(self.emis_list)] = value  # delete pm2.5 input

            dataX.append(dataXi)
            dataY.append(dataYi)
            del dataXi, dataYi

        return np.array(dataX), np.array(dataY)


'''

sample generator for danet:

    - no shuffle in this function, should shuffle the training and validation filenames
      before call this function

      include 2 freatures: prediction (AOD550 included) and observation (satellite)

      input: prediction; satellite-prediction
      output: reanalysis - prediction

'''


class sample_generator_danet(keras.utils.Sequence):

    def __init__(self, path_pred, path_obs, path_refer, indices, batch_size, nx, ny, dic_norm, n_feature=2):
        self.path_pred = path_pred  # path for model prediciton
        self.path_obs = path_obs  # path for satellite data
        self.path_refer = path_refer  # path for reference (label)
        self.file_list = indices
        self.batch_size = batch_size
        self.nx = nx
        self.ny = ny
        self.dic_norm = dic_norm
        self.n_feature = n_feature

    #  computes the number of batches that this generator is supposed to produce
    def __len__(self):
        return int(math.ceil(len(self.file_list) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.file_list[idx * self.batch_size: (idx + 1) * self.batch_size]

        dataX = []
        dataY = []

        for idxi in batch_idx:
            # inxi: filename of forecast file
            # organise one sample
            dataXi = np.zeros((1, self.nx, self.ny, self.n_feature), dtype='float32')
            dataYi = np.zeros((1, self.nx, self.ny), dtype='float32')

            # load model prediciton
            # filename = self.inpath1.format(idxi.strftime("%Y%m%d%H"))
            filename = idxi
            dataXi[..., 0] = np.load(filename)[...,1]  # wrong simulation: AOD only

            # extract datatime string
            match = re.search(r'(\d{10})\.npy', os.path.basename(idxi))
            if match:
                timestamp = match.group(1)
            else:
                print("Timestamp not found in the file path.")

            # load satellite observation
            filename = self.path_obs.format(timestamp)
            data_temp = np.load(filename)
            # data_temp = np.nan_to_num(data_temp, nan=0.0)
            data_temp = np.roll(data_temp, 240, axis=1)
            mask = np.isnan(data_temp)
            data_temp[mask] = dataXi[0, ..., 0][mask]
            dataXi[0, ..., -1] = data_temp

            # reference (label)
            filename = self.path_refer.format(timestamp)
            dataYi[0, ...] = np.load(filename)[..., 5]  # for PM2.5 and AOD550

            # difference - AOD
            dataXi[0, ..., 1] = dataXi[0, ..., 1] - dataXi[0, ..., 0]
            dataYi[0, ...] = dataYi[0, ...] - dataXi[0, ..., 0]

            dataX.append(dataXi)
            dataY.append(dataYi)

            del dataXi, dataYi

        # normalization
        dataX = np.array(dataX)
        dataY = np.array(dataY)

        key = 'aod550'
        amin = self.dic_norm[key][0]
        amax = self.dic_norm[key][1]
        dataX[..., 0] = (dataX[..., 0] - amin) / (amax - amin)
        # AOD difference
        key = 'aod550_diff'
        amin = self.dic_norm[key][0]
        amax = self.dic_norm[key][1]
        dataX[..., 1] = (dataX[..., 1] - amin) / (amax - amin)
        dataY = (dataY - amin) / (amax - amin)

        return np.array(dataX), np.array(dataY)