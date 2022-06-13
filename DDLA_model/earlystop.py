# -*- coding: utf-8 -*-
"""
Our work with Early Stopping

@author: Donggeun Kwon (donggeun.kwon@gmail.com) 
"""

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling1D, Activation
from tensorflow.keras.layers import Input, MaxPooling1D, Conv1D, BatchNormalization, Add
from tensorflow.keras.initializers import glorot_normal, he_normal

import numpy as np
import tensorflow as tf
import os, time, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import utils.hyperparameters as para


def labeling(pt, k, mode='LSB'):
    if mode=='LSB':
        output_size = 2
        label = np.array([np.eye(output_size)[(para.SBOX[int(p) ^ k]) & 1] for p in pt])
    elif mode=='MSB':
        output_size = 2
        label = np.array([np.eye(output_size)[(para.SBOX[int(p) ^ k]) >> 7] for p in pt])
    else:
        raise NameError('Labeling must be "MSB" or "LSB".')
        
    return label, output_size


def attack(trace, pt):
    # Input/Output Shape 
    input_size = np.shape(trace)[1]
    model_name = 'DDLA_ES.h5'
    
    label_list = []
    output_size = 2
    
    for k in range(para.KEY_SIZE):
        # Labeling
        label,_ = labeling(pt, k, 'LSB')
        label_list.append(label)
        
        # MLP_{exp}
        inputLy = Input(shape=(input_size, ))
        hidden1 = Dense(20, activation='relu')(inputLy)
        batchn1 = BatchNormalization()(hidden1)
        hidden2 = Dense(10, activation='relu')(batchn1)
        batchn2 = BatchNormalization()(hidden2)
        outputly = Dense(output_size, activation='softmax')(batchn2)
            
        # Build
        model = Model(inputLy, outputly)
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=para.LEARNING_RATE), 
                      metrics=['accuracy']) 
        model.save('./es_model/'+str(k)+'_'+model_name)
        K.clear_session()
        
    # Train
    for e in range(para.EPOCH): 
        for k in range(para.KEY_SIZE):
            model = load_model('./es_model/'+str(k)+'_'+model_name)
            model.fit(trace, label_list[k], 
                      epochs=1, 
                      batch_size=para.BATCH_SIZE, 
                      verbose=0)
            model.save('./es_model/'+str(k)+'_'+model_name)
            K.clear_session()
    

    return None