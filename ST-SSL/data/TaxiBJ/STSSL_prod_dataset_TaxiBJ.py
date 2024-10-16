import h5py
import numpy as np
import scipy
import random
import os
import pandas as pd




input_matrix_num = 8
test_num = 485
all_need_matrix_num = 4000+input_matrix_num

f = h5py.File('C:\\XRT\\HNU\\Dataset\\交通流量\\TaxiBJ\\BJ13_M32x32_T30_InOut.h5')
mat = f['data'][:all_need_matrix_num]  # (Long, 2, 32, 32)
Long, Ch, H, W = mat.shape
mat = mat.reshape([Long, 2, H*W])
mat = np.transpose(mat, [0, 2, 1])

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


