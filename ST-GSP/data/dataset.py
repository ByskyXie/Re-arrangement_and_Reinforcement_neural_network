import sys
sys.path.append("./")
import numpy as np
import h5py
import os
import math
import torch
import torch.utils.data as data
from data.external import external_taxibj, external_bikenyc
from data.minmax_normalization import MinMaxNormalization
from data.data_fetcher import DataFetcher
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
import random
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


def get_traffic_matrix_video(path, all_batch_size=1000, name: str = 'Video_matrix'):
    files = os.listdir(path)  # list current path files
    # filter other file
    index = len(files) - 1
    while index >= 0:
        if files[index].find(name) == -1:
            del (files[index])
        index -= 1

    # sort file
    files.sort()  

    assert len(files) >= all_batch_size
    files = files[:all_batch_size]
    tms = []  # traffic matrix

    print(f'Begin load {name}')
    for timestamp in files:
        tm = np.loadtxt(path + '/' + timestamp, dtype=np.float)
        tms.append(tm)
    print('Data loaded')

    tms = np.stack(tms)  # [all_batch, w, h] 

    return tms, torch.Tensor(list(range(len(tms))))


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

    return tms, torch.Tensor(list(range(len(tms))))


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
    return tms, torch.Tensor(list(range(len(tms))))


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

        if fn_get_traffic_matrix is None:  # get data matrix fn
            fn_get_traffic_matrix = get_traffic_matrix_abilene

        all_need_matrix_num =all_batch_num+self.input_matrix_num+self.predict_matrix_num\
                             +max(self.len_trend, self.period_interval)
        if sampling_rate2 is not None and sampling_rate2!=1:
            all_need_matrix_num = int(all_need_matrix_num/sampling_rate2)
        tms, time_seq = fn_get_traffic_matrix(path, all_need_matrix_num)

        self.time_seq = time_seq
        self.tms = tms if type(tms) is np.ndarray else torch.from_numpy(tms)

        # resort row and column
        # print('Row&Column rearranging...')
        # self.tms = self.rearrange_matrix_row_and_column(self.tms)

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
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        shape = self.tms.shape
        # MinMaxScaler only accepts dim <= 2, so the data is reshaped and appled
        self.scaler.fit(self.tms.reshape(-1, 1))
        self.tms = torch.from_numpy(self.scaler.fit_transform(self.tms.reshape(-1, 1)))
        self.tms = self.tms.reshape(shape)


        if gpu_mode:
            # if np.prod(self.tms.shape) < 1e8:
                self.large_dataset_flag = False
                self.tms = self.tms.to(torch.device('cuda:0'))
                self.masks = self.masks.to(torch.device('cuda:0'))
            # else:
            #     # Cause hard to sending full dataset into GPU
            #     self.large_dataset_flag = True

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
        time_seq = self.time_seq[indices].to(torch.float32)

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

        # train_close = torch.unsqueeze(train_close, dim=0).to(torch.float32)
        # train_trend = torch.unsqueeze(train_trend, dim=0).to(torch.float32)
        # train_period = torch.unsqueeze(train_period, dim=0).to(torch.float32)
        # valid = torch.unsqueeze(valid, dim=0).to(torch.float32)

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
            time_seq = time_seq.to(torch.device('cuda:0'))

        train = torch.cat([train_close, train_trend, train_period], dim=0)
        sum_len = self.len_trend+self.len_period+self.input_matrix_num
        train_ext = torch.zeros([sum_len, 1])
        valid_ext = torch.zeros([1])

        return [train, train_ext, valid, valid_ext]

    def __len__(self):
        return self.all_batch_num


class Dataset:
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, dconf, test_days=-1, datapath=datapath):
        self.dconf = dconf
        self.dataset = dconf.name
        self.datapath = datapath
        if self.dataset == 'TaxiBJ':
            self.datafolder = 'TaxiBJ'
            self.dataname = [
                'BJ13_M32x32_T30_InOut.h5',
                'BJ14_M32x32_T30_InOut.h5',
                'BJ15_M32x32_T30_InOut.h5',
                'BJ16_M32x32_T30_InOut.h5'
            ]
            self.nb_flow = 2
            self.dim_h = 32
            self.dim_w = 32
            self.T = 48
            test_days = 28 if test_days == -1 else test_days

            self.external_builder = external_taxibj(
                self.datapath,
                fourty_eight=dconf.fourty_eight,
                previous_meteorol=dconf.previous_meteorol
            )

        elif self.dataset == 'BikeNYC':
            self.datafolder = 'BikeNYC'
            self.dataname = ['NYC14_M16x8_T60_NewEnd.h5']
            self.nb_flow = 2
            self.dim_h = 16
            self.dim_w = 8
            self.T = 24
            test_days = 10 if test_days == -1 else test_days

            self.external_builder = external_bikenyc()

        else:
            raise ValueError('Invalid dataset')

        print("test_days:", test_days)
        self.len_test = test_days * self.T
        self.portion = dconf.portion

    def get_raw_data(self):
        raw_data_list = list()
        raw_ts_list = list()
        print("  Dataset: ", self.datafolder)
        for filename in self.dataname:
            f = h5py.File(os.path.join(self.datapath, self.datafolder, filename), 'r')
            _raw_data = f['data'][()]
            _raw_ts = f['date'][()]
            f.close()

            raw_data_list.append(_raw_data)
            raw_ts_list.append(_raw_ts)

        return raw_data_list, raw_ts_list

    @staticmethod
    def remove_incomplete_days(data, timestamps, t=48):
        print("before removing", len(data))
        days = []
        days_incomplete = []
        i = 0
        while i < len(timestamps):
            if int(timestamps[i][8:]) != 1:
                i += 1
            elif i + t - 1 < len(timestamps) and int(timestamps[i + t - 1][8:]) == t:
                days.append(timestamps[i][:8])
                i += t
            else:
                days_incomplete.append(timestamps[i][:8])
                i += 1
        print("incomplete days: ", days_incomplete)
        days = set(days)
        idx = []
        for i, t in enumerate(timestamps):
            if t[:8] in days:
                idx.append(i)

        data = data[idx]
        timestamps = [timestamps[i] for i in idx]
        print("after removing", len(data))
        return data, timestamps

    def trainset_of(self, vec):
        return vec[:math.floor((len(vec)-self.len_test) * self.portion)]

    def testset_of(self, vec):
        return vec[-math.floor(self.len_test * self.portion):]

    def split(self, x, y):
        x_train = self.trainset_of(x)
        x_test = self.testset_of(x)
        y_train = self.trainset_of(y)
        y_test = self.testset_of(y)

        return x_train, y_train, x_test, y_test

    def load_data(self):
        print('Preprocessing: Reading HDF5 file(s)')
        raw_data_list, ts_list = self.get_raw_data()

        data_list, ts_new_list = [], []
        for idx in range(len(ts_list)):
            raw_data = raw_data_list[idx]
            ts = ts_list[idx]

            if self.dconf.rm_incomplete_flag:
                raw_data, ts = self.remove_incomplete_days(raw_data, ts, self.T)

            data_list.append(raw_data)
            ts_new_list.append(ts)

        raw_data = np.concatenate(data_list)

        print('Preprocessing: Min max normalizing')
        mmn = MinMaxNormalization()
        train_dat = self.trainset_of(raw_data)
        mmn.fit(train_dat)
        new_data_list = [
            mmn.transform(data).astype('float32', copy=False)
            for data in data_list
        ]

        x_list, y_list, ts_x_list, ts_y_list = [], [], [], []
        for idx in range(len(ts_new_list)):
            x, y, ts_x, ts_y = \
                DataFetcher(new_data_list[idx], ts_new_list[idx], self.T).fetch_data(self.dconf)
            x_list.append(x)
            y_list.append(y)
            ts_x_list.append(ts_x)
            ts_y_list.append(ts_y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        ts_x = np.concatenate(ts_x_list)
        ts_y = np.concatenate(ts_y_list)

        x_train, y_train, x_test, y_test = self.split(x, y)
        if self.dconf.ext_flag:
            ext_x, ext_y = self.external_builder(ts_x, ts_y, timeenc=self.dconf.timeenc_flag)
            ext_x_train, ext_y_train, ext_x_test, ext_y_test = self.split(ext_x, ext_y)
        else:
            ext_x_train, ext_y_train, ext_x_test, ext_y_test = None, None, None, None

        class TempClass:
            def __init__(self_2):
                self_2.X_train = x_train
                self_2.Y_train = y_train
                self_2.X_test = x_test
                self_2.Y_test = y_test
                self_2.ext_X_train = ext_x_train
                self_2.ext_Y_train = ext_y_train
                self_2.ext_X_test = ext_x_test
                self_2.ext_Y_test = ext_y_test
                self_2.mmn = mmn
                self_2.ts_Y_train = self.trainset_of(ts_y)
                self_2.ts_Y_test = self.testset_of(ts_y)
            
            def show(self_2):
                print(
                    "Run: X inputs shape: ", self_2.X_train.shape, self_2.X_test.shape,
                    "Y inputs shape: ", self_2.Y_train.shape, self_2.Y_test.shape
                )
                print("Run: min~max: ", self_2.mmn.min, '~', self_2.mmn.max)
                if self.dconf.ext_flag:
                    print(
                        "Run: X-ext inputs' shapes:", self_2.ext_X_train.shape, self_2.ext_X_test.shape,
                        "Y-ext inputs' shapes:", self_2.ext_Y_train.shape, self_2.ext_Y_test.shape
                    )
        return TempClass()

class TorchDataset(data.Dataset):
    def __init__(self, ds, mode='train', select_pre=0):
        super(TorchDataset, self).__init__()
        self.ds = ds
        self.mode = mode
        self.select_pre = select_pre
        self.len = int(self.ds.X_train[0].shape[0]/2)+1
    
    def __getitem__(self, index):
        if self.mode == 'train':
            x_all = np.concatenate((self.ds.Y_train[index], self.ds.X_train[index]),axis=0)
            x_ext_all = np.concatenate((np.expand_dims(self.ds.ext_Y_train[index],0), self.ds.ext_X_train[index]),axis=0)
            tmp = np.split(x_all,self.len,axis=0)
            x_all_delete_index = [self.select_pre*2, self.select_pre*2+1]
            x_ext_all_delete_index = [self.select_pre]
            x_train_new = np.delete(x_all, x_all_delete_index, 0)
            x_ext_new = np.delete(x_ext_all, x_ext_all_delete_index, 0)
            X = torch.from_numpy(x_train_new)
            X_ext = torch.from_numpy(x_ext_new)
            Y = torch.from_numpy(tmp[self.select_pre])
            Y_ext = torch.from_numpy(x_ext_all[self.select_pre])
        else:
            x_all = np.concatenate((self.ds.Y_test[index], self.ds.X_test[index]),axis=0)
            x_ext_all = np.concatenate((np.expand_dims(self.ds.ext_Y_test[index],0), self.ds.ext_X_test[index]),axis=0)
            tmp = np.split(x_all,self.len,axis=0)
            x_all_delete_index = [self.select_pre*2, self.select_pre*2+1]
            x_ext_all_delete_index = [self.select_pre]
            x_train_new = np.delete(x_all, x_all_delete_index, 0)
            x_ext_new = np.delete(x_ext_all, x_ext_all_delete_index, 0)
            X = torch.from_numpy(x_train_new)
            X_ext = torch.from_numpy(x_ext_new)
            Y = torch.from_numpy(tmp[self.select_pre])
            Y_ext = torch.from_numpy(x_ext_all[self.select_pre])

        return X.float(), X_ext.float(), Y.float(), Y_ext.float()

    def __len__(self):
        if self.mode == 'train':
            return self.ds.X_train.shape[0]
        else:
            return self.ds.X_test.shape[0]

class DatasetFactory(object):
    def __init__(self, dconf):
        self.dataset = Dataset(dconf)
        self.ds = self.dataset.load_data()

    def get_train_dataset(self, select_pre):
        return TorchDataset(self.ds, 'train', select_pre)
    
    def get_test_dataset(self, select_pre):
        return TorchDataset(self.ds, 'test', select_pre)

if __name__ == '__main__':
    class DataConfiguration:
        # Data
        name = 'TaxiBJ'
        portion = 1.  # portion of data

        len_close = 3
        len_period = 1
        len_trend = 1
        pad_forward_period = 0
        pad_back_period = 0
        pad_forward_trend = 0
        pad_back_trend = 0

        len_all_close = len_close * 1
        len_all_period = len_period * (1 + pad_back_period + pad_forward_period)
        len_all_trend = len_trend * (1 + pad_back_trend + pad_forward_trend)
           
        len_seq = len_all_close + len_all_period + len_all_trend
        cpt = [len_all_close, len_all_period, len_all_trend]

        interval_period = 1
        interval_trend = 7

        ext_flag = True
        timeenc_flag = 'w'
        rm_incomplete_flag = True
        fourty_eight = True
        previous_meteorol = True

    df = DatasetFactory(DataConfiguration())
    ds = df.get_train_dataset(0)
    print(ds.ds.show())
    X, X_ext, Y, Y_ext = next(iter(ds))
    print('train:')
    print(X.size())
    print(X_ext.size())
    print(Y.size())
    print(Y_ext.size())
    