# -*- coding: utf-8 -*-
"""
Main code

@author: Donggeun Kwon (donggeun.kwon@gmail.com) 
"""

from memory_profiler import profile # https://pypi.org/project/memory-profiler
import time
import os, sys, h5py
import numpy as np

mode = ['timon', 'earlystop', 'parallel', 'sharedlayer', 'cnn_tm', 'cnn_sl', 'cnn_pl']
MODE_USED = mode[0] # ***select the model you want***

exec('import DDLA_model.' + MODE_USED + ' as DDLA')

# if you want more speed, turn off memory_profiler
# @profile
def attack(trace, label):
    DDLA.attack(trace, label)

if __name__ == '__main__':
    # Data Load
    start = time.time() # Start
    
    # ***input your dataset***
    tmp = h5py.File('SCA_Dataset.h5', 'r') 
    trace = np.array(tmp['trace'])
    pt = np.reshape(np.array(tmp['pt']), [-1])
    
    # Simple pre-processing
    trace = np.double(trace)
    # trace = (trace - np.mean(trace, 0))
    trace = (trace - np.mean(trace))
    trace = (trace) / (np.max(np.abs(trace)))
    
    print('####################################### Attack: '+MODE_USED)
    attack(trace, pt)
    print(MODE_USED + ' ' +"time :", time.time() - start) #End
    print('####################################### End')