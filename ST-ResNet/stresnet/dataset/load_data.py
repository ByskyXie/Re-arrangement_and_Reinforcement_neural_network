from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

import torch
import random
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import numpy as np
import tables
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler

current_dir = Path(__file__).resolve().parent


def get_traffic_matrix_brain375(path: str = '.', all_batch_size=1000):
    data = np.load(path)
    tms = data['matrices']
    # date = data['date']

    assert len(tms) >= all_batch_size
    tms = tms[:all_batch_size].astype(np.float32)
    # date = date[:all_batch_size]

    # tms = tms[...,:160,:160]

    # tms = np.clip(tms, 0, 1e7)
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
        return [train_close, train_trend, train_period, torch.Tensor([0]), valid]

    def __len__(self):
        return self.all_batch_num



@dataclass
class LoadData:
    data_files: list
    holiday_file: str
    meteorology_file: str

    def __post_init__(self):
        self.DATA_PATHS = []
        for data_file in self.data_files:
            data_path = Path(data_file)
            if data_path.exists():
                self.DATA_PATHS.append(data_path)
            else:
                raise FileNotFoundError

        holiday_path = Path(self.holiday_file)
        if holiday_path.exists():
            self.HOLIDAY = holiday_path
        else:
            self.HOLIDAY = None

        meteorology_path = Path(self.meteorology_file)
        if meteorology_path.exists():
            self.METEOROLOGY = meteorology_path
        else:
            self.METEOROLOGY = None

    def load_holiday(self, timeslots: np.ndarray) -> Optional[np.ndarray]:
        if not self.HOLIDAY:
            return None

        timeslots = np.frompyfunc(lambda x: x[:8], 1, 1)(timeslots)

        with open(self.HOLIDAY, "r") as f:
            holidays = set([l.strip() for l in f])

        holidays = np.array(holidays)
        indices = np.where(np.isin(timeslots, holidays))[0]

        hv = np.zeros(len(timeslots))
        hv[indices] = 1.0
        return hv[:, np.newaxis]

    def load_meteorol(self, timeslots: np.ndarray):
        dat = tables.open_file(self.METEOROLOGY, mode="r")
        # dateformat: YYYYMMDD[slice]
        m_timeslots = dat.root.date.read().astype(str)
        wind_speed = dat.root.WindSpeed.read()
        weather = dat.root.Weather.read()
        temperature = dat.root.Temperature.read()
        dat.close()

        predicted_ids = np.where(np.isin(m_timeslots, timeslots))[0]
        cur_ids = predicted_ids - 1

        ws = wind_speed[cur_ids]
        wr = weather[cur_ids]
        te = temperature[cur_ids]

        # 0-1 scale
        ws = minmax_scale(ws)[:, np.newaxis]
        te = minmax_scale(te)[:, np.newaxis]

        # concatenate all these attributes
        merge_data = np.hstack([wr, ws, te])

        return merge_data

    def _remove_incomplete_days(
        self, dat: tables.file.File, T: int):  # -> tuple[np.ndarray, np.ndarray]
        # 20140425 has 24 timestamps, which does not appear in `incomplete_days` in the original implementation.
        # So I reimplemented it in a different way.
        data = dat.root.data.read()
        timestamps = dat.root.date.read().astype(str)

        dates, values = np.vstack(
            np.frompyfunc(lambda x: (x[:8], x[8:]), 1, 2)(timestamps)
        )
        # label encoding
        uniq_dates, labels = np.unique(dates, return_inverse=True)
        # groupby("labels")["values"].sum() != sum(range(1, 49))
        incomplete_days = uniq_dates[
            np.where(np.bincount(labels, values.astype(int)) != sum(range(1, (T + 1))))[
                0
            ]
        ]
        del_idx = np.where(np.isin(dates, incomplete_days))[0]
        new_data = np.delete(data, del_idx, axis=0)
        new_timestamps = np.delete(timestamps, del_idx)
        return new_data, new_timestamps

    def load_data(self, T: int):  # -> tuple[list[np.ndarray], list[np.ndarray]]:
        data_all = []
        timestamp_all = []
        for data_path in self.DATA_PATHS:
            dat = tables.open_file(data_path, mode="r")
            data, timestamps = self._remove_incomplete_days(dat, T=T)
            data[data < 0] = 0.0
            data_all.append(data)
            timestamp_all.append(timestamps)
            dat.close()

        return data_all, timestamp_all
