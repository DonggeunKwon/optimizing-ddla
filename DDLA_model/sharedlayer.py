# -*- coding: utf-8 -*-
"""
Our work with Shared layers

@author: Donggeun Kwon (donggeun.kwon@gmail.com) 
"""

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling1D, Activation
from tensorflow.keras.layers import Input, MaxPooling1D, Conv1D, BatchNormalization, Add, Concatenate
from tensorflow.keras.initializers import glorot_normal, he_normal

import numpy as np
import tensorflow as tf
import os, time, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import utils.hyperparameters as para


def labeling_all(pt):
    label = []
    
    for k in range(para.KEY_SIZE):
        label.append(np.array([[(para.SBOX[int(p) ^ k]) & 1] for p in pt]).reshape([-1]))
        
    return np.array(label).T

def attack(trace, pt):
    input_size = np.shape(trace)[1]
    # Labeling
    label = labeling_all(pt)
    
    # MLP
    input_layer = Input(shape=(input_size, ))   
    
    hidden_layer_1_fc=(Dense(20, activation='relu')(input_layer))
    hidden_layer_1_bn=(BatchNormalization()(hidden_layer_1_fc))   
    hidden_layer_2_fc=(Dense(10, activation='relu')(hidden_layer_1_bn))
    hidden_layer_2_bn=(BatchNormalization()(hidden_layer_2_fc))
    output_layer=(Dense(para.KEY_SIZE, activation='sigmoid')(hidden_layer_2_bn))
    
    # for i in range(para.KEY_SIZE):
    #   concat_layer = Concatenate(axis=-1)(output_layer);
    
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