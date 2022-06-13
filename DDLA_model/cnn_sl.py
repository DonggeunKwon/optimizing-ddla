# -*- coding: utf-8 -*-
"""
Our work with Shared layers

@author: Donggeun Kwon (donggeun.kwon@gmail.com) 
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
    
    hidden_layer_1_fc=(Conv1D(filters=4, kernel_size=32, padding='same')(input_layer))
    hidden_layer_1_bn=(BatchNormalization()(hidden_layer_1_fc))
    hidden_layer_1_pl=(AveragePooling1D(pool_size=2)(hidden_layer_1_bn))
    hidden_layer_1_at=(Activation('relu')(hidden_layer_1_pl))
    
    hidden_layer_2_fc=(Conv1D(filters=4, kernel_size=16, padding='same')(hidden_layer_1_at))
    hidden_layer_2_bn=(BatchNormalization()(hidden_layer_2_fc))
    hidden_layer_2_pl=(AveragePooling1D(pool_size=4)(hidden_layer_2_bn))
    hidden_layer_2_at=(Activation('relu')(hidden_layer_2_pl))
    
    flatten_layer=Flatten()(hidden_layer_2_at)
    
    output_layer=(Dense(para.KEY_SIZE, activation='sigmoid')(flatten_layer))
    
    # Bulid
    model = Model(input_layer, output_layer)
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(lr=para.LEARNING_RATE)) 
    # model.summary()
    # Train
    for e in range(para.EPOCH): 
        model.fit(trace, label, 
                  epochs=1, 
                  batch_size=para.BATCH_SIZE, 
                  verbose=0)

    return None