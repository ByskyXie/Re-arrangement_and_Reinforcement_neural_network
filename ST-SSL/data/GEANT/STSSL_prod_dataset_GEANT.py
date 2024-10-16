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

    temp = np.array(tms)  # [all_batch, 24, 24]  : B,R,C
    x, y = np.split(temp, [1], 1)
    x, tms = np.split(y, [1], 2)  # cause 23*23 MCSTL model hard processing

    tms = np.clip(tms, 0, 2e6)  # filter anomaly value
    # tms = np.array(tms)  # [all_batch, 24, 24]  : B,R,C
    # tms = np.expand_dims(tms, axis=1)  # [all_batch, 1, 24, 24] : B,Ch,R,C
    return tms


input_matrix_num = 8
test_num = 485
all_need_matrix_num = 4000+input_matrix_num

mat = get_traffic_matrix_geant("C:\\Python project\\DataFillingCPC\\GEANT\\", all_need_matrix_num)

L = mat.shape[0]
all_idx = list(range(input_matrix_num, L-1))
random.shuffle(all_idx)
train_num = int(0.9*(len(all_idx)))
train_idx = all_idx[:train_num]
test_idx = all_idx[train_num:]
# select_idx = np.random.rand(test_num)
# select_idx = select_idx*(L-input_matrix_num) + input_matrix_num
# select_idx = select_idx.astype(np.int32)


def save_dataset(mat, select_idx, dir, name):
    y = mat[select_idx]  # (B, R, C)
    y = np.expand_dims(y, axis=1)

    if not os.path.exists(dir):
        os.makedirs(dir)
    print("DIR:", dir)

    x = []
    for idx in select_idx:
        # input X
        X = mat[idx-input_matrix_num: idx]  # [HistorySeq, R*C, 1]
        x.append(X)

    x = np.stack(x)  # [B, HistorySeq, R, C]

    # x = np.concatenate([x, x], axis=-1)
    # y = np.concatenate([y, y], axis=-1)
    np.savez(dir+f'{name}', x=x, y=y)  # Y: as label
    print("输出标签Y，shape=", x.shape, y.shape)




save_dataset(mat, train_idx, dir='./', name='train.npz')
save_dataset(mat, test_idx, dir='./', name='val.npz')
save_dataset(mat, test_idx, dir='./', name='test.npz')


