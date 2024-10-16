import numpy as np
from Param_DSTN_flow import *
import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import random
import torch
from sklearn.preprocessing import MinMaxScaler

def getXSYS(data):
    interval_p, interval_t = 1, 7  # day interval for period/trend
    depends = [range(1, len_closeness + 1),
               [interval_p * DAYTIMESTEP * i for i in range(1, len_period + 1)],
               [interval_t * DAYTIMESTEP * i for i in range(1, len_trend + 1)]]

    start = max(len_closeness, interval_p * DAYTIMESTEP * len_period, interval_t * DAYTIMESTEP * len_trend)
    end = data.shape[0]

    XC, XP, XT = [], [], []
    for i in range(start, end):
        x_c = [data[i - j] for j in depends[0]]
        x_p = [data[i - j] for j in depends[1]]
        x_t = [data[i - j] for j in depends[2]]
        XC.append(np.concatenate(x_c, axis=0))
        XP.append(np.concatenate(x_p, axis=0))
        XT.append(np.concatenate(x_t, axis=0))
    XC, XP, XT = np.array(XC), np.array(XP), np.array(XT)
    # XS = [XC, XP, XT, day_feature] if day_feature is not None else [XC, XP, XT]
    YS = data[start:end]
    return XC, XP, XT, YS

def getXSYSFour(mode, data_all):
    testNum = int((4848 + 4368 + 5520 + 6624) * (1 - trainRatio))
    XC_train, XP_train, XT_train, YS_train = [], [], [], []
    XC_test, XP_test, XT_test, YS_test = [], [], [], []

    for i in range(3):
        data = data_all[i]
        XC, XP, XT, YS = getXSYS(data)
        XC_train.append(XC)
        XP_train.append(XP)
        XT_train.append(XT)
        YS_train.append(YS)
    for i in range(3,4):
        data = data_all[i]
        XC, XP, XT, YS = getXSYS(data)
        XC_train.append(XC[:-testNum])
        XP_train.append(XP[:-testNum])
        XT_train.append(XT[:-testNum])
        YS_train.append(YS[:-testNum])
        XC_test.append(XC[-testNum:])
        XP_test.append(XP[-testNum:])
        XT_test.append(XT[-testNum:])
        YS_test.append(YS[-testNum:])
    XC_train = np.vstack(XC_train)
    XP_train = np.vstack(XP_train)
    XT_train = np.vstack(XT_train)
    YS_train = np.vstack(YS_train)
    XC_test = np.vstack(XC_test)
    XP_test = np.vstack(XP_test)
    XT_test = np.vstack(XT_test)
    YS_test = np.vstack(YS_test)
    if mode == 'train':
        return XC_train, XP_train, XT_train, YS_train
    elif mode == 'test':
        return XC_test, XP_test, XT_test, YS_test
    else:
        assert False, 'invalid mode...'
        return None

def preload(dataFile_lst):
    process_ratio = 0.14
    data_all = []
    for item in dataFile_lst:
        data = np.load(item)
        data = data.transpose((0, 3, 1, 2))
        print(item, data.shape)
        data = data[:int(len(data)*process_ratio)]
        data_all.append(data)
    return data_all

def load_data():
    data = preload(dataFile_lst)
    data_norm = [x / MAX_FLOWIO for x in data]
    XC_train, XP_train, XT_train, YS_train = getXSYSFour('train', data_norm)
    print(XC_train.shape, XP_train.shape, XT_train.shape, YS_train.shape)
    XC_test, XP_test, XT_test, YS_test = getXSYSFour('test', data_norm)
    print(XC_test.shape, XP_test.shape, XT_test.shape, YS_test.shape)
    XS_train = np.concatenate((XC_train, XP_train, XT_train), axis=1)
    XS_test = np.concatenate((XC_test, XP_test, XT_test), axis=1)
    return XS_train, YS_train, XS_test, YS_test


def get_traffic_matrix_geant(path: str = '.', all_batch_size=1000):
    files = os.listdir(path)  # list current path files
    # filter other file
    index = len(files) - 1
    while index >= 0:
        if files[index].find('IntraTM') == -1:
            del (files[index])
        index -= 1
    # sort file
    files.sort()  

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

    temp = np.array(tms)  # [all_batch, 24, 24] 
    x, y = np.split(temp, [1], 1)  
    x, tms = np.split(y, [1], 2)  

    # tms = np.concatenate([tms, tms], axis=-1)  
    # tms = np.concatenate([tms, tms], axis=-2)

    tms = np.clip(tms, 0, 2e6) # filter anomaly value

    return tms, np.array(list(range(len(tms))))


def get_traffic_matrix_abilene(path: str = '.', all_batch_size=1000):
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

    tms = np.array(tms)  # [all_batch, 12, 12] 
    return tms, np.array(list(range(len(tms))))


class MatrixPredDataset(Dataset):
    large_dataset_flag = False

    def __init__(self, path, all_batch_num, fn_get_traffic_matrix=None, predict_matrix_num=3, input_matrix_num=4
                 , gpu_mode=False, sampling_rate1=None, sampling_rate2=None, len_period=1, len_trend=1
                 , period_interval=12, trend_interval=2):
        assert sampling_rate2 is None or (sampling_rate2 > 0 and sampling_rate2 <= 1)
        assert sampling_rate1 is None or (sampling_rate1 > 0 and sampling_rate1 <= 1)

        self.all_batch_num = all_batch_num
        self.predict_matrix_num = predict_matrix_num
        self.input_matrix_num = input_matrix_num
        self.gpu_mode = gpu_mode
        self.sampling_rate1 = sampling_rate1  # sampling rate of matrix
        self.sampling_rate2 = sampling_rate2  # sampling rate of time interval
        self.len_period = len_period
        self.len_trend = len_trend
        self.period_interval = period_interval
        self.trend_interval = trend_interval
        self.scaler = None

        if fn_get_traffic_matrix is None:  # get data matrix fn
            fn_get_traffic_matrix = get_traffic_matrix_abilene

        all_need_matrix_num =all_batch_num+self.input_matrix_num+self.predict_matrix_num\
                             +max(self.len_trend, self.period_interval)
        if sampling_rate2 is not None and sampling_rate2!=1:
            all_need_matrix_num = int(all_need_matrix_num/sampling_rate2)
        tms, time_seq = fn_get_traffic_matrix(path, all_need_matrix_num)

        self.time_seq = time_seq
        self.tms = tms if type(tms) is np.ndarray else torch.from_numpy(tms)

        self.tms = torch.from_numpy(self.tms)

        if self.sampling_rate2 is not None and self.sampling_rate2 < 1:
            # should delete some matrix
            indices = list(range(len(self.tms)))
            random.shuffle(indices)  # 打乱序号
            indices = indices[:int(self.sampling_rate2*len(self.tms))]  # clip tms
            indices.sort()
            self.tms = self.tms[indices]
            self.time_seq = self.time_seq[indices]

        print(f'sampling_rate1={sampling_rate1}\tsampling_rate2={sampling_rate2}')
        print(f'max={self.tms.max()} min={self.tms.min()} mean={self.tms.mean()} shape={self.tms.shape}')
        if sampling_rate1 is not None:
            self.masks = self.get_masks(self.tms)

        # scaler data
        # self.row = self.tms.shape[-2]
        # self.column = self.tms.shape[-1]
        # self.scaler = MinMaxScaler(feature_range=(0, 1))
        # shape = self.tms.shape
        # # MinMaxScaler only accepts dim <= 2, so the data is reshaped and appled
        # self.scaler.fit(self.tms.reshape(-1, self.row*self.column))
        # self.tms = torch.from_numpy(self.scaler.fit_transform(self.tms.reshape(-1, self.row*self.column)))
        # self.tms = self.tms.reshape(shape)

    def get_masks(self, tms):
        masks = []
        for i in tms:
            masks.append(self.produce_a_mask(i))
        masks = torch.stack(masks)
        masks = masks.type(torch.FloatTensor)

        if self.gpu_mode and self.large_dataset_flag:
            masks = masks.to(torch.device('cuda:0'))
        return masks

    def produce_a_mask(self, matrix):
        amount = int(torch.prod(torch.from_numpy(np.array(matrix.shape))))
        one_num = int(amount * self.sampling_rate1)
        zero_num = amount - one_num
        mask = np.concatenate([np.ones(one_num), np.zeros(zero_num)])
        np.random.shuffle(mask)
        return torch.from_numpy(mask.reshape(matrix.shape))

    def __getitem__(self, index):
        offset = max(self.len_trend, self.period_interval)
        index += offset  # cause this method use period info

        # train cloase data
        head_matrix_pos = index
        tail_matrix_pos = index+self.input_matrix_num

        indices = list(range(head_matrix_pos, tail_matrix_pos))
        train_close = self.tms[indices]

        if self.sampling_rate1 is not None and self.sampling_rate1!=1:  # partial sampling
            train_close_mask = self.masks[head_matrix_pos: tail_matrix_pos]
            train_close = train_close * train_close_mask

        # train trend
        assert index - self.len_trend >= 0
        train_trend = self.tms[[index - self.len_trend]]
        # train period
        assert index - self.period_interval >= 0
        train_period = self.tms[[index - self.period_interval]]

        # valid
        head_valid_pos = index+self.input_matrix_num+1
        tail_valid_pos = head_valid_pos+self.predict_matrix_num
        # head_valid_pos = (index+self.input_matrix_num-1)*(1+self.predict_matrix_num)+1  # for prediction
        # tail_valid_pos = (index+self.input_matrix_num-1)*(1+self.predict_matrix_num)+1+self.predict_matrix_num # for prediction
        valid = self.tms[head_valid_pos: tail_valid_pos]

        train_close = train_close.to(torch.float32)
        train_trend = train_trend.to(torch.float32)
        train_period = train_period.to(torch.float32)
        valid = valid.to(torch.float32)

        if self.gpu_mode and self.large_dataset_flag:
            # Cause hard to sending full dataset into GPU
            train_close = train_close.to(torch.device('cuda:0'))
            train_trend = train_trend.to(torch.device('cuda:0'))
            train_period = train_period.to(torch.device('cuda:0'))
            valid = valid.to(torch.device('cuda:0'))

        train = torch.cat([train_close, train_period, train_trend], dim=0)

        return [train, valid]

    def __len__(self):
        return self.all_batch_num

    def load_pred_data(self, train_ratio: float = 0.8):
        assert train_ratio<=1 and train_ratio>0
        XS_train, YS_train, XS_test, YS_test = [], [], [], []
        indices = list(range(self.__len__()))
        # random.shuffle(indices)
        limitation = int(self.__len__() * train_ratio)

        for idx in range(self.__len__()):
            train, valid = self.__getitem__(idx)
            if idx < limitation:
                XS_train.append(train)
                YS_train.append(valid)
            else:
                XS_test.append(train)
                YS_test.append(valid)
        XS_train, YS_train = np.stack(XS_train, axis=0), np.stack(YS_train, axis=0)
        XS_test, YS_test = np.stack(XS_test, axis=0), np.stack(YS_test, axis=0)
        return XS_train, YS_train, XS_test, YS_test


def main():
    XS_train, YS_train, XS_test, YS_test = load_data()
    print(type(XS_train), type(YS_train), type(XS_test), type(YS_test))
    print(XS_train.shape, YS_train.shape, XS_test.shape, YS_test.shape)

if __name__ == '__main__':
    # main()
    datasets = MatrixPredDataset(path='Abilene_dataset_path', all_batch_num=2000,
                                 predict_matrix_num=1,
                                 input_matrix_num=len_closeness,
                                 fn_get_traffic_matrix=get_traffic_matrix_abilene, gpu_mode=True,
                                 sampling_rate1=1, sampling_rate2=1, len_period=len_period, len_trend=len_trend
                                 , period_interval=T_period, trend_interval=T_trend)
    XS_train, YS_train, XS_test, YS_test = datasets.load_pred_data(train_ratio=trainRatio)
    del datasets
    print(type(XS_train), type(YS_train), type(XS_test), type(YS_test))
    print(XS_train.shape, YS_train.shape, XS_test.shape, YS_test.shape)

    # datasets = MatrixPredDataset(path='GEANT_dataset_path', all_batch_num=4000,
    #                              predict_matrix_num=1,
    #                              input_matrix_num=8,
    #                              fn_get_traffic_matrix=get_traffic_matrix_geant, gpu_mode=True,
    #                              sampling_rate1=1, sampling_rate2=1)
