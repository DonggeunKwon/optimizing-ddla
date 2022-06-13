# -*- coding: utf-8 -*-
"""
Timon's DDLA

@author: Donggeun Kwon (donggeun.kwon@gmail.com)
"""

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling1D, Activation
from tensorflow.keras.layers import Input, MaxPooling1D, Conv1D, BatchNormalization, Add, AveragePooling1D
from tensorflow.keras.initializers import glorot_normal, he_normal

import numpy as np
import tensorflow as tf
import os, time, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import utils.hyperparameters as para


def labeling(pt, k):
    output_size = 2
    label = np.array([np.eye(output_size)[(para.SBOX[int(p) ^ k]) & 1] for p in pt])
        
    return label, output_size


def attack(trace, pt):
    # Input/Output Shape 
    input_size = np.shape(trace)[1]
    trace = trace[:, :, np.newaxis]
    
    for k in range(para.KEY_SIZE):
        # Labeling
        label, output_size = labeling(pt, k)
        
        # MLP_{exp}
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
        
        output_layer=(Dense(output_size, activation='softmax')(flatten_layer))
            
        # Build
        model = Model(input_layer, output_layer)
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=para.LEARNING_RATE), 
                      metrics=['accuracy']) 
        # model.summary()
        # Train
        for e in range(para.EPOCH): 
            model.fit(trace, label, 
                      epochs=1, 
                      batch_size=para.BATCH_SIZE, 
                      verbose=0)
    
        K.clear_session()

    return None