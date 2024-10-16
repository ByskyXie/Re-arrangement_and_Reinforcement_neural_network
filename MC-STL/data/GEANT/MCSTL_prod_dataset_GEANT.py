import h5py
import numpy as np
import scipy
import random
import os
import pandas as pd
import xml.etree.ElementTree as ET


def get_traffic_matrix_geant(path: str = '.', all_batch_size=1000):
    """
        read in dataset GEANT from files.
    """
    files = os.listdir(path)  # list current path files
    # filter other file
    index = len(files) - 1
    while index >= 0:
        if files[index].find('IntraTM') == -1:
            del (files[index])
        index -= 1
    # sort file
    files.sort()  # sort by timestamp

    assert len(files) >= all_batch_size
    files = files[:all_batch_size]
    tms = []  # traffic matrix

    print('Begin load GEANT')
    # 解析xml file
    tree = ET.ElementTree()
    for timestamp in files:
        tm = [[0.0 for j in range(24)] for i in range(24)]
        ele = tree.parse(path + '/' + timestamp)
        for row in ele[1]:
            row_id = int(row.get('id'))
            for node in row:
                col_id = int(node.get('id'))
                tm[row_id][col_id] = float(node.text)
        tms.append(tm)
    print('GEANT loaded')

    # temp = np.array(tms)  # [all_batch, 24, 24]  : B,R,C
    # x, y = np.split(temp, [1], 1)
    # x, tms = np.split(y, [1], 2)  # cause 23*23 MCSTL model hard processing


    tms = np.array(tms)  # [all_batch, 24, 24]  : B,R,C
    tms = np.clip(tms, 0, 2e6) # filter anomaly value
    tms = np.expand_dims(tms, axis=1)  # [all_batch, 1, 24, 24] : B,Ch,R,C
    return tms



input_matrix_num = 8
test_num = 485
all_need_matrix_num = 4000+input_matrix_num

mat = get_traffic_matrix_geant("C:\\Python project\\DataFillingCPC\\GEANT\\", all_need_matrix_num)
np.clip(mat, 0, 1e6) ########### clip anomaly
_max=1
_min=0
mat = (mat-_min)/_max  # normalization

# save mm.npy
class CustomArray:
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __getitem__(self, key):
        if key == 'min':
            return self.min_value
        elif key == 'max':
            return self.max_value
        else:
            raise KeyError("Invalid key. Use 'min' or 'max'.")

mm = np.array({min: _min, max: _max}, dtype=object)
np.save("mm.npy", mm)

Long = mat.shape[0]
all_idx = list(range(input_matrix_num, Long-1))
random.shuffle(all_idx)
train_num = int(0.9*(len(all_idx)))
train_idx = all_idx[:train_num]
test_idx = all_idx[train_num:]
# select_idx = np.random.rand(test_num)
# select_idx = select_idx*(Long-input_matrix_num) + input_matrix_num
# select_idx = select_idx.astype(np.int32)


def save_dataset(mat, select_idx, dir):
    basis = mat[select_idx]  # (test_num, 2, 32, 32)

    if not os.path.exists(dir):
        os.makedirs(dir)
    print("DIR:", dir)

    np.save(dir+'basis.npy', basis)                                                 # Y: as label
    print("输出标签Y，shape=", basis.shape)


    time_correlation = []
    for idx in select_idx:
        # input X
        X = mat[idx-input_matrix_num: idx]  # [HistorySeq, W, H]
        time_correlation.append(X)

    time_correlation = np.stack(time_correlation)
    np.save(dir+'time_correlation.npy', time_correlation)                           # X: as input
    print("输出输入X，shape=", time_correlation.shape)



save_dataset(mat, train_idx, dir='./train/')
save_dataset(mat, test_idx, dir='./test/')


