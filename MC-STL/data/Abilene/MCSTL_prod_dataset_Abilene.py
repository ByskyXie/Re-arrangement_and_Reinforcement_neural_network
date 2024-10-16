import h5py
import numpy as np
import scipy
import random
import os
import pandas as pd


def get_traffic_matrix_abilene(path: str = '.', all_batch_size=1000):
    """
        read in dataset abilene from files.
    """
    files = os.listdir(path)  # list current path files
    # filter other file
    index = len(files) - 1
    while index >= 0:
        if files[index].find('tm.2004') == -1:
            del (files[index])
        index -= 1

    # sort file
    files.sort()

    assert len(files) >= all_batch_size
    files = files[:all_batch_size]
    tms = []  # traffic matrix

    print('Begin load Abilene')
    for timestamp in files:
        tm = []
        with open(path + '/' + timestamp) as file:
            while True:
                line = file.readline()
                if line == '':
                    break
                if line[0] == '#':
                    continue
                line = line.strip().split(',')
                tm.append([float(num) for num in line])
        tms.append(tm)
    print('Data loaded')

    tms = np.array(tms)  # [all_batch, 12, 12] : B,R,C
    tms = np.expand_dims(tms, axis=1)  # [all_batch, 1, 12, 12] : B,Ch,R,C
    return tms


input_matrix_num = 8
test_num = 485
all_need_matrix_num = 4000+input_matrix_num

mat = get_traffic_matrix_abilene("C:\\Python project\\DataFillingCPC\\Measured\\", all_need_matrix_num)
_max=mat.max()
_min=mat.min()
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


