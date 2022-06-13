# Optimizing DDLA
Implementations of improved DDLA

Refer:
Optimizing Implementations of Non-Profiled Deep Learning-Based Side-Channel Attacks [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9669913)]

Contact: <donggeun.kwon@gmail.com>


### Intro
In this paper, we propose three methods to solve the aforementioned problems and any challenges resulting from solving these problems. First, we propose a modified algorithm that allows the monitoring of the metrics in the intermediate process. Second, we propose a parallel neural network architecture and algorithm that works by training a single network, with no need to re-train the same model repeatedly. Attacks were performed faster with the proposed algorithm than with the previous algorithm. Finally, we propose a novel architecture that uses shared layers to solve memory problems in parallel architecture and also helps achieve better performance.

### Source Code
The structure of the source code is as follows:

* main.py
* /DDLA_model
    + timon.py
    + earlystop.py
    + parallel.py
    + sharedlayer.py
    + cnn_tm.py (_Timon's CNNDDLA_)
    + cnn_sl.py (CNN with sharedlayer)
    + cnn_pl.py (CNN with parallel)
    + /utils
      + hyperparameters.py

## **Notice**
Source code modifications are required.

### main.py

1. Select the DDLA model.
```python
mode = ['timon', 'earlystop', 'parallel', 'sharedlayer', 'cnn_tm', 'cnn_sl', 'cnn_pl']
MODE_USED = mode[...] # 0 ~ 6
```
2. Input the dataset.
```python
# input your dataset
# tmp = h5py.File('SCA_Dataset.h5', 'r')
trace = ...
pt = ...
```

### hyperparameters.py
```python
BATCH_SIZE = ...
LEARNING_RATE = ...
EPOCH = ...
```

<center><a href="http://crypto.korea.ac.kr" target="_blank"><img src="http://crypto.korea.ac.kr/wp-content/uploads/2019/01/Algorithm_trans.png" width="30%" height="30%" /></a></center>