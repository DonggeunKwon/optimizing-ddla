# -*- coding: utf-8 -*-
"""
Our work with Parallel

@author: @author: Donggeun Kwon (donggeun.kwon@gmail.com) 
"""

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling1D, Activation, AveragePooling1D
from tensorflow.keras.layers import Input, MaxPooling1D, Conv1D, BatchNormalization, Add, Concatenate
from tensorflow.keras.initializers import glorot_normal, he_normal

import numpy as np
import tensorflow as tf
import os, time, sys
'''
import scipy
import scipy.io
from datetime import datetime
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import utils.hyperparameters as para


def labeling_all(pt):
    label = []
    
    for k in range(para.KEY_SIZE):
        label.append(np.array([[(para.SBOX[int(p) ^ k]) & 1] for p in pt]).reshape([-1]))
        
    return np.array(label).T

def attack(trace, pt):
    trace = trace[:, :, np.newaxis]
    
    input_size = np.shape(trace)[1]
    # Labeling
    label = labeling_all(pt)
    
    # MLP
    input_layer = Input(shape=(input_size, 1))  
    
    hidden_layer_1_cn = []
    hidden_layer_1_pl = []
    hidden_layer_1_bn = []
    hidden_layer_2_cn = []
    hidden_layer_2_pl = []
    hidden_layer_2_bn = []
    flatten_layer = []
    output_layer = []
       
    
    for i in range(256):
        hidden_layer_1_cn.append(Conv1D(filters=4, kernel_size=32, activation='relu')(input_layer))
        hidden_layer_1_bn.append(BatchNormalization()(hidden_layer_1_cn[-1]))
        hidden_layer_1_pl.append(AveragePooling1D(pool_size=2)(hidden_layer_1_bn[-1]))
        hidden_layer_2_cn.append(Conv1D(filters=4, kernel_size=16, activation='relu')(hidden_layer_1_pl[-1]))
        hidden_layer_2_bn.append(BatchNormalization()(hidden_layer_2_cn[-1]))
        hidden_layer_2_pl.append(AveragePooling1D(pool_size=4)(hidden_layer_2_bn[-1]))            
        flatten_layer.append(Flatten()(hidden_layer_2_pl[-1]))
        output_layer.append(Dense(1, activation='sigmoid')(flatten_layer[-1]))
    
    concat_layer = Concatenate(axis=-1)(output_layer);
    
    model = Model(input_layer, concat_layer)
    
    # Bulid
    model = Model(input_layer, concat_layer)
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(lr=para.LEARNING_RATE)) 
    
    # Train
    for e in range(para.EPOCH): 
        model.fit(trace, label, 
                  epochs=1, 
                  batch_size=para.BATCH_SIZE, 
                  verbose=0)
        
    return None