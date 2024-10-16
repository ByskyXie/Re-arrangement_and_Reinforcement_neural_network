import h5py
import numpy as np
import scipy
import random
import os
import pandas as pd

input_matrix_num = 8
all_need_matrix_num = 4000+input_matrix_num
max=1230
min=0

f = h5py.File('BJ13_M32x32_T30_InOut.h5')
holiday_list = np.loadtxt("BJ_Holiday.txt").astype(np.int32)  # shape=(106,)
Meteorology = h5py.File("BJ_Meteorology.h5")
M_Temperature = np.array(Meteorology['Temperature']).astype(np.float32)  # shape=(59006,)
M_Weather = np.array(Meteorology['Weather']).astype(np.int32)  # shape=(59006, 17)
M_WindSpeed = np.array(Meteorology['WindSpeed']).astype(np.float32)   # shape=(59006,)
M_date = np.array(Meteorology['date']).astype(np.int32)  # shape=(59006,)

mat=f['data'][:all_need_matrix_num]  # (Long, 2, 32, 32)
mat = np.array(mat)  # Dataset --> np.array
mat = (mat-min)/max  # normalization

date = f['date'][:all_need_matrix_num] # shape=(Long)  item=b'2013070101' item_type=<class 'numpy.bytes_'>
date = np.array(date)  # Dataset --> np.array
date = date.astype(np.int32)
# get weekend list
date_series = pd.to_datetime(date); day_of_week = date_series.dayofweek;
is_weekend = (day_of_week >= 5).astype(np.int32)


Long = mat.shape[0]
all_idx = list(range(input_matrix_num, Long-1))
random.shuffle(all_idx)
train_num = int(0.9*(len(all_idx)))
train_idx = all_idx[:train_num]
test_idx = all_idx[train_num:]
# select_idx = np.random.rand(test_num)
# select_idx = select_idx*(Long-input_matrix_num) + input_matrix_num
# select_idx = select_idx.astype(np.int32)


def save_dataset(mat, date, select_idx, dir):
    basis = mat[select_idx]  # (test_num, 2, 32, 32)

    if not os.path.exists(dir):
        os.makedirs(dir)
    print("DIR:", dir)

    np.save(dir+'basis.npy', basis)                                                 # Y: as label
    print("输出标签Y，shape=", basis.shape)

    # input X_ext
    X_date = date[select_idx]  # [HistorySeq, 2013070101]
    feature_0 = X_date // 100 % 100  # Day feature
    feature_1 = is_weekend[select_idx]  # embed_weekend feature
    feature_2 = X_date % 100  # embed_hour feature
    feature_3 = (np.array([D // 100 in holiday_list for D in X_date], dtype=np.int32))  # embed_holiday feature

    pos_idx = [M_date.tolist().index(D) for D in X_date]
    feature_4 = M_WindSpeed[pos_idx]  # embed_wind feature
    feature_5 = np.argmax(M_Weather[pos_idx], axis=1)  # embed_weather feature
    feature_6 = M_Temperature[pos_idx]  # embed_temperature feature

    time_c_feature = np.stack([feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6])
    time_c_feature = np.transpose(time_c_feature, [1, 0])

    np.save(dir+'time_c_feature.npy', time_c_feature)                               # X_ext: as input of time
    print("输出输入X_ext，shape=", time_c_feature.shape)


    time_correlation = []
    for idx in select_idx:
        # input X
        X = mat[idx-input_matrix_num: idx]  # [HistorySeq, W, H]
        time_correlation.append(X)

    time_correlation = np.stack(time_correlation)
    np.save(dir+'time_correlation.npy', time_correlation)                           # X: as input
    print("输出输入X，shape=", time_correlation.shape)



save_dataset(mat, date, train_idx, dir='./train/')
save_dataset(mat, date, test_idx, dir='./test/')


