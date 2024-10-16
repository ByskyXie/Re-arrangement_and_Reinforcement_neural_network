from sa_convlstm import SAConvLSTM
import torch
import os
import torch.nn as nn
from config import configs
from config_for_taxibj import configs_forBJ
from config_for_abilene import configs_forAB
from config_for_geant import configs_forGT
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import h5py
from utils import *
import math
import random
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import random_split
import time
import xml.etree.ElementTree as ET
from sklearn.preprocessing import MinMaxScaler


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(5)
        self.network = SAConvLSTM(configs.input_dim, configs.hidden_dim, configs.d_attn, configs.kernel_size,
                                  configs.matrix_row, configs.matrix_column).to(configs.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.3, patience=0, verbose=False, min_lr=0.0001)
        self.weight = torch.from_numpy(np.array([1.5]*4 + [2]*7 + [3]*7 + [4]*6) * np.log(np.arange(24)+1)).to(configs.device)

    def score(self, y_pred, y_true):
        with torch.no_grad():
            sc = score(y_pred, y_true, self.weight)
        return sc.item()

    def loss_sst(self, y_pred, y_true):
        # y_pred/y_true (N, 37, 24, 48)
        rmse = torch.mean((y_pred - y_true)**2, dim=[2, 3])
        rmse = torch.sum(rmse.sqrt().mean(dim=0))
        return rmse

    def loss_nino(self, y_pred, y_true):
        with torch.no_grad():
            rmse = torch.sqrt(torch.mean((y_pred - y_true)**2, dim=0))  # * self.weight  # previous
        return rmse.sum()

    def train_once(self, sst, nino_true, ratio):  # unsqeeze ---> [:, :, None]
        sst_pred, nino_pred = self.network(sst.float()[:, :, None], teacher_forcing=True,
                                           scheduled_sampling_ratio=ratio, train=True,
                                           input_frames=self.configs.input_length,
                                           future_frames=self.configs.output_length,
                                           output_frames=self.configs.input_length+self.configs.output_length-1)
        self.optimizer.zero_grad()
        loss_sst = self.loss_sst(sst_pred, sst[:, 1:].to(self.device))
        loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
        loss_sst.backward()
        if configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
        self.optimizer.step()
        return loss_sst.item(), loss_nino.item(), nino_pred

    def test(self, dataloader_test):
        nino_pred = []
        sst_pred = []
        with torch.no_grad():
            for sst, _ in dataloader_test:
                sst, nino = self.network(sst.float()[:, :12, None], teacher_forcing=False, train=False,
                                           input_frames=self.configs.input_length,
                                           future_frames=self.configs.output_length,
                                           output_frames=self.configs.output_length)
                nino_pred.append(nino)
                sst_pred.append(sst)

        return torch.cat(sst_pred, dim=0), torch.cat(nino_pred, dim=0)

    def infer(self, dataset, dataloader, datasets=None):
        # calculate loss_func and score on a eval/test set
        self.network.eval()
        rmse, er, nmae, counter = 0, 0, 0, 0
        loss_sst, loss_nino = 0, 0
        with torch.no_grad():
            for j, (sst, nino_true) in enumerate(dataloader):
                # combine train&valid as SaConvLSTM train input
                sst = torch.cat([sst, nino_true], dim=1)
                # [:, :-1, None] is because cannot introduce GT
                sst_pred, nino_pred = self.network(sst.float()[:, :-self.configs.output_length, None]
                                                   , teacher_forcing=False, train=False,
                                                   input_frames=self.configs.input_length,
                                                   future_frames=self.configs.output_length,
                                                   output_frames=self.configs.output_length)

                nino_true = nino_true
                sst_true = sst[:, -self.configs.output_length:]
                # sc = self.score(nino_pred, nino_true)
                # print(sst_pred.shape, sst_true.shape, '000000000000000000')
                loss_sst += self.loss_sst(sst_pred, sst_true).item()
                loss_nino += self.loss_nino(nino_pred, nino_true).item()

                trues = nino_true.cpu().detach()
                nino_pred = nino_pred.cpu().detach()
                if datasets is not None and datasets.scaler is not None:
                    shape1 = trues.shape
                    shape2 = nino_pred.shape
                    trues = datasets.scaler.inverse_transform(trues.reshape(-1, datasets.row * datasets.column)).reshape(shape1)
                    nino_pred = datasets.scaler.inverse_transform(nino_pred.reshape(-1, datasets.row * datasets.column)).reshape(shape2)
                    trues = torch.from_numpy(trues)
                    nino_pred = torch.from_numpy(nino_pred)
                rmse += RMSE(trues, nino_pred)
                er += error_rate(trues, nino_pred)
                nmae += NMAE(trues, nino_pred)
                counter += 1

        return loss_sst/counter, loss_nino/counter, rmse/counter, er/counter, nmae/counter

    def train(self, dataset_train, dataset_eval, chk_path, datasets=None):
        torch.manual_seed(0)
        print('loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        print('loading eval dataloader')
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)

        count = 0
        best = math.inf
        ssr_ratio = 1
        for i in range(self.configs.num_epochs):
            # print('\nepoch: {0}'.format(i+1))
            # if i < 40:  # if matrix size is too large to convergence, you can training task model in begining 40 epochs, therefore training R&E NN
            #     self.network.arranger.requires_grad_(False)
            # else:
            #     self.network.arranger.requires_grad_(True)
            begin_time = time.time()
            self.network.train()
            for j, (sst, nino_true) in enumerate(dataloader_train):
                if ssr_ratio > 0:
                    ssr_ratio = max(ssr_ratio - self.configs.ssr_decay_rate, 0)
                # combine train&valid as SaConvLSTM train input
                sst = torch.cat([sst, nino_true], dim=1)
                loss_sst, loss_nino, nino_pred = self.train_once(sst, nino_true, ssr_ratio)

                # if j % self.configs.display_interval == 0:
                #     trues = nino_true.float().to(self.device)
                #     rmse = RMSE(trues, nino_pred)
                #     er = error_rate(trues, nino_pred)
                #     nmae = NMAE(trues, nino_pred)
                #     # sc = self.score(nino_pred, nino_true.float().to(self.device))
                #     print(f'ER_train=\t{er}\tNMAE=\t{nmae}\tRMSE=\t{rmse}\tbatch_training_loss: {loss_sst}, {loss_nino}')

            # evaluation
            loss_sst_eval, loss_nino_eval, sc_eval, er, nmae = self.infer(dataset=dataset_eval, dataloader=dataloader_eval, datasets=datasets)
            rmse = sc_eval
            print(f'Epoch [{i}]: ER_eval=\t{er}\tNMAE=\t{nmae}\tRMSE=\t{rmse}\tTime=\t{(time.time() - begin_time)}\t'
                  f'batch_training_loss: {loss_sst_eval}, {loss_nino_eval}')
            self.lr_scheduler.step(sc_eval)
            if sc_eval >= best:
                count += 1
                # print('eval score is not improved for {} epoch'.format(count))
            else:
                count = 0
                # print('eval score is improved from {:.5f} to {:.5f}, saving model'.format(best, sc_eval))
                self.save_model(chk_path)
                best = sc_eval

            if self.configs.early_stopping and count == self.configs.patience:
                # print('early stopping reached, best score is {:5f}'.format(best))
                break

    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)


def prepare_data(ds_dir):  # return train_ipt, train_GT, eval_ipt, eval_GT, test_ipt, test_GT
    # train/eval/test split
    cmip6sst, cmip5sst, cmip6nino, cmip5nino = read_raw_data(ds_dir)
    # if the processed data has been stored
    # cmip6sst, cmip5sst, cmip6nino, cmip5nino = read_from_nc(ds_dir)
    sst_train = [cmip6sst, cmip5sst[..., :-2]]  # train_ipt
    nino_train = [cmip6nino, cmip5nino[..., :-2]]  # train_GT
    sst_eval = [cmip5sst[..., -2:-1]]  # eval_ipt
    nino_eval = [cmip5nino[..., -2:-1]]  # eval_GT
    sst_test = [cmip5sst[..., -1:]]  # test_ipt
    nino_test = [cmip5nino[..., -1:]]  # test_GT
    return sst_train, nino_train, sst_eval, nino_eval, sst_test, nino_test

# dataset detail :https://tianchi.aliyun.com/competition/entrance/531871/information
# label: Neno3.4 SST异常指数，数据维度为（year,month）
# 测试用的初始场（输入）数据为国际多个海洋资料同化结果提供的随机抽取的n段12个时间序列.
#   数据格式采用NPY格式保存，维度为（12，lat，lon, 4）,12为t时刻及过去11个时刻，4为预测因子，并按照SST,T300,Ua,Va的顺序存放。
#   测试集文件序列的命名规则：test_编号_起始月份_终止月份.npy，如test_00001_01_12_.npy。
# Test set:
# Input size: (12, 24, 72, 4) | 12是时间 | 24(lat)为 | 4为预测因子(SST,T300,Ua,Va)
# Label size: (24,)


def train_for_neno():
    print(configs.__dict__)

    print('\nreading data')
    sst_train, nino_train, sst_eval, nino_eval, sst_test, nino_test = prepare_data('tcdata/enso_round1_train_20210201')

    print('processing training set')
    dataset_train = cmip_dataset(sst_train[0], nino_train[0], sst_train[1], nino_train[1], samples_gap=10)
    print(dataset_train.GetDataShape())
    del sst_train
    del nino_train
    print('processing eval set')
    dataset_eval = cmip_dataset(sst_cmip6=None, nino_cmip6=None,
                                sst_cmip5=sst_eval[0], nino_cmip5=nino_eval[0], samples_gap=5)
    print(dataset_eval.GetDataShape())
    del sst_eval
    del nino_eval
    trainer = Trainer(configs)
    trainer.save_configs('config_train.pkl')
    trainer.train(dataset_train, dataset_eval, 'checkpoint.chk')
    print('\n----- training finished -----\n')

    del dataset_train
    del dataset_eval

    print('processing test set')
    dataset_test = cmip_dataset(sst_cmip6=None, nino_cmip6=None,
                                sst_cmip5=sst_test[0], nino_cmip5=nino_test[0], samples_gap=1)
    print(dataset_test.GetDataShape())

    # test
    print('loading test dataloader')
    dataloader_test = DataLoader(dataset_test, batch_size=configs.batch_size_test, shuffle=False)
    chk = torch.load('checkpoint.chk')
    trainer.network.load_state_dict(chk['net'])
    print('testing...')
    loss_sst_test, loss_nino_test, sc_test, er, nmae = trainer.infer(dataset=dataset_test, dataloader=dataloader_test)
    print('test loss:\n sst: {:.2f}, nino: {:.2f}, score: {:.4f}'.format(loss_sst_test, loss_nino_test, sc_test))



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

    # 去除读入时，行列index=0时的边缘
    temp = np.array(tms)  # [all_batch, 24, 24]
    x, y = np.split(temp, [1], 1)
    x, tms = np.split(y, [1], 2)

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


def get_traffic_matrix_taxibj(path: str = '.', all_batch_size=1000, InData=True):
    print('Begin load TaxiBj')
    f = h5py.File(path, 'r')
    print(f'Keys contains data&date, use InData:{InData}')
    tms = f['data'][:, 0, :, :] if InData else f['data'][:, 1, :, :]
    print('TaxiBj loaded')

    assert len(tms) >= all_batch_size
    tms = tms[:all_batch_size]
    return tms, torch.Tensor(list(range(len(tms))))


class MatrixDataset(Dataset):
    large_dataset_flag = False

    def __init__(self, path, all_batch_num, fn_get_traffic_matrix=None, predict_matrix_num=3, input_matrix_num=4
                 , gpu_mode=False, sampling_rate1=None, sampling_rate2=None):
        assert sampling_rate2 is None or (sampling_rate2 > 0 and sampling_rate2 <= 1)
        assert sampling_rate1 is None or (sampling_rate1 > 0 and sampling_rate1 <= 1)

        self.all_batch_num = all_batch_num
        self.predict_matrix_num = predict_matrix_num
        self.input_matrix_num = input_matrix_num
        self.gpu_mode = gpu_mode
        self.sampling_rate1 = sampling_rate1  # sampling rate of matrix
        self.sampling_rate2 = sampling_rate2  # sampling rate of time interval
        self.scaler = None

        if fn_get_traffic_matrix is None:  # get data matrix fn
            fn_get_traffic_matrix = get_traffic_matrix_taxibj

        all_need_matrix_num = (all_batch_num+self.input_matrix_num)*(1+self.predict_matrix_num)
        if sampling_rate2 is not None and sampling_rate2 < 1:
            all_need_matrix_num = int(all_need_matrix_num/sampling_rate2)
        # get raw data
        tms, time_seq = fn_get_traffic_matrix(path, all_need_matrix_num)
        # PE1: trans time to position embed
        # time_seq = pd.to_datetime(time_seq)
        # time_seq = pd.DataFrame(time_seq, columns=['date'])
        # # 'freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h'
        # time_seq = time_features(time_seq, timeenc=0, freq='h')
        # # PE2
        time_seq = pd.to_datetime(time_seq)
        time_seq -= pd.to_datetime('Jul  1 1900, 00:00:00')
        time_seq = pd.Series(time_seq).dt.total_seconds()
        time_seq = torch.tensor(time_seq).numpy()

        self.time_seq = time_seq
        self.tms = tms if type(tms) is np.ndarray else torch.from_numpy(tms)


        self.tms = torch.from_numpy(self.tms)
        self.time_seq = torch.from_numpy(self.time_seq)

        if self.sampling_rate2 is not None and self.sampling_rate2 < 1:
            # should delete some matrix
            indices = list(range(len(self.tms)))
            random.shuffle(indices)
            indices = indices[:int(self.sampling_rate2*len(self.tms))]  # clip tms
            indices.sort()
            self.tms = self.tms[indices]
            self.time_seq = self.time_seq[indices]

        print(f'sampling_rate1={sampling_rate1}\tsampling_rate2={sampling_rate2}')
        print(f'max={self.tms.max()} min={self.tms.min()} mean={self.tms.mean()} shape={self.tms.shape}')

        # scaler data
        self.row = self.tms.shape[-2]
        self.column = self.tms.shape[-1]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        shape = self.tms.shape
        # MinMaxScaler only accepts dim <= 2, so the data is reshaped and appled
        self.scaler.fit(self.tms.reshape(-1, self.row*self.column))
        self.tms = torch.from_numpy(self.scaler.fit_transform(self.tms.reshape(-1, self.row*self.column)))
        self.tms = self.tms.reshape(shape)

        if sampling_rate1 is not None:
            self.masks = self.get_masks(self.tms)
        if gpu_mode:
            # if np.prod(self.tms.shape) < 1e8:
                self.large_dataset_flag = False
                self.tms = self.tms.to(torch.device('cuda:0'))
                self.time_seq = self.time_seq.to(torch.device('cuda:0'))
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
        # train data
        head_matrix_pos = index
        tail_matrix_pos = index + self.input_matrix_num

        indices = list(range(head_matrix_pos, tail_matrix_pos))
        train = self.tms[indices]
        time_seq = self.time_seq[indices].to(torch.float32)

        if self.sampling_rate1 is not None and self.sampling_rate1 != 1:  # partial sampling
            train_mask = self.masks[head_matrix_pos: tail_matrix_pos]
            train = train * train_mask

        # valid
        head_valid_pos = index + self.input_matrix_num + 1
        tail_valid_pos = head_valid_pos + self.predict_matrix_num
        # head_valid_pos = (index+self.input_matrix_num-1)*(1+self.predict_matrix_num)+1  # for prediction
        # tail_valid_pos = (index+self.input_matrix_num-1)*(1+self.predict_matrix_num)+1+self.predict_matrix_num # for prediction
        valid = self.tms[head_valid_pos: tail_valid_pos]

        # train = torch.unsqueeze(train, dim=0).to(torch.float32)
        # valid = torch.unsqueeze(valid, dim=0).to(torch.float32)

        if self.gpu_mode and self.large_dataset_flag:
            # Cause hard to sending full dataset into GPU
            train = train.to(torch.device('cuda:0'))
            valid = valid.to(torch.device('cuda:0'))
            time_seq = time_seq.to(torch.device('cuda:0'))
        assert train.shape[0] == self.input_matrix_num
        return [train, valid]

    def __len__(self):
        return self.all_batch_num


def get_not_zero_position(inputs):
    return torch.clamp(torch.clamp(torch.abs(inputs), 0, 1e-32) * 1e36, 0, 1)

def NMAE(source, target):
    """ should mul mask previous """
    source = source.reshape(-1)
    target = target.reshape(-1)
    molecular = torch.abs(source - target)
    denominator = torch.abs(target)
    not_zero_pos = get_not_zero_position(target)
    return torch.sum(not_zero_pos * molecular) / torch.sum(denominator)

def RMSE(source, target):
    """ should mul mask previous """
    source = source.reshape(-1)
    target = target.reshape(-1)
    not_zero_pos = get_not_zero_position(target)

    return torch.sqrt(torch.pow(not_zero_pos*(source-target), 2).mean())

def error_rate(source, target):
    """ should mul mask previous """
    source = source.reshape(-1)
    target = target.reshape(-1)
    molecular = torch.pow(source - target, 2)
    denominator = torch.pow(target, 2)
    not_zero_pos = get_not_zero_position(target)
    return torch.pow(torch.sum(not_zero_pos * molecular) / torch.sum(denominator), 1 / 2)


def train_for_taxibj():
    print(configs_forBJ.__dict__)

    print('\nreading data')
    # load taxiBJ
    datasets = MatrixDataset(path='TaxiBJ_dataset_path', all_batch_num=1250,
                             predict_matrix_num=configs_forBJ.predict_matrix_num,
                             input_matrix_num=configs_forBJ.input_matrix_num,
                             fn_get_traffic_matrix=get_traffic_matrix_taxibj, gpu_mode=True,
                             sampling_rate1=configs_forBJ.sampling_rate1, sampling_rate2=configs_forBJ.sampling_rate2)
    trains, develops, tests = random_split(datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)),
                                                      round(0.1 * len(datasets))],
                                           generator=torch.Generator().manual_seed(23))
    dataset_train = trains
    dataset_eval = develops

    trainer = Trainer(configs_forBJ)

    # load RE weights
    # if configs_forBJ.load_RE_weight:
    #     trainer.network.arranger.load("./arranger_taxi_STresnet_dict.pth")
    if configs_forBJ.RE_trainable is False:
        trainer.network.arranger.requires_grad_(False)

    # load weight
    print('----------- load weight ------------')
    chk = torch.load('checkpoint.chk')
    trainer.network.load_state_dict(chk['net'])

    trainer.save_configs('config_train.pkl')
    trainer.train(dataset_train, dataset_eval, 'checkpoint.chk')
    print('\n----- training finished -----\n')

    del dataset_train, dataset_eval, trains, develops

    # test
    dataset_test = tests

    print('loading test dataloader')
    dataloader_test = DataLoader(dataset_test, batch_size=configs_forBJ.batch_size_test, shuffle=False)
    chk = torch.load('checkpoint.chk')
    trainer.network.load_state_dict(chk['net'])
    print('testing...')
    begin_time = time.time()
    loss_sst_test, loss_nino_test, sc_test, er, nmae = trainer.infer(dataset=dataset_test, dataloader=dataloader_test)
    rmse = sc_test
    print(f'ER_eval=\t{er}\tNMAE=\t{nmae}\tRMSE=\t{rmse}\tTime=\t{(time.time() - begin_time)}\t'
          f'batch_training_loss: {loss_sst_test}, {loss_nino_test}')


def train_for_abilene():
    print(configs_forAB.__dict__)

    print('\nreading data')
    # load abilene
    datasets = MatrixDataset(path='Abilene_dataset_path', all_batch_num=4000,
                             predict_matrix_num=configs_forAB.predict_matrix_num,
                             input_matrix_num=configs_forAB.input_matrix_num,
                             fn_get_traffic_matrix=get_traffic_matrix_abilene, gpu_mode=True,
                             sampling_rate1=configs_forAB.sampling_rate1, sampling_rate2=configs_forAB.sampling_rate2)
    trains, develops, tests = random_split(datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)),
                                                      round(0.1 * len(datasets))],
                                           generator=torch.Generator().manual_seed(23))
    dataset_train = trains
    dataset_eval = develops

    trainer = Trainer(configs_forAB)

    # load RE weights
    # if configs_forAB.load_RE_weight:
    #     trainer.network.arranger.load("./arranger_taxi_STresnet_dict.pth")
    if configs_forAB.RE_trainable is False:
        trainer.network.arranger.requires_grad_(False)

    # load weight
    # print('----------- load weight ------------')
    # chk = torch.load('checkpoint.chk')
    # trainer.network.load_state_dict(chk['net'])

    trainer.save_configs('config_train.pkl')
    trainer.train(dataset_train, dataset_eval, 'checkpoint.chk', datasets)
    print('\n----- training finished -----\n')

    del dataset_train, dataset_eval, trains, develops

    # test
    dataset_test = tests

    print('loading test dataloader')
    dataloader_test = DataLoader(dataset_test, batch_size=configs_forAB.batch_size_test, shuffle=False)
    chk = torch.load('checkpoint.chk')
    trainer.network.load_state_dict(chk['net'])
    print('testing...')
    begin_time = time.time()
    loss_sst_test, loss_nino_test, sc_test, er, nmae = trainer.infer(dataset=dataset_test, dataloader=dataloader_test)
    rmse = sc_test
    print(f'ER_eval=\t{er}\tNMAE=\t{nmae}\tRMSE=\t{rmse}\tTime=\t{(time.time() - begin_time)}\t'
          f'batch_training_loss: {loss_sst_test}, {loss_nino_test}')


def train_for_geant():
    print(configs_forGT.__dict__)

    print('\nreading data')
    # load abilene
    datasets = MatrixDataset(path='GEANT_dataset_path', all_batch_num=4000,
                             predict_matrix_num=configs_forGT.predict_matrix_num,
                             input_matrix_num=configs_forGT.input_matrix_num,
                             fn_get_traffic_matrix=get_traffic_matrix_geant, gpu_mode=True,
                             sampling_rate1=configs_forGT.sampling_rate1, sampling_rate2=configs_forGT.sampling_rate2)
    trains, develops, tests = random_split(datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)),
                                                      round(0.1 * len(datasets))],
                                           generator=torch.Generator().manual_seed(23))
    dataset_train = trains
    dataset_eval = develops

    trainer = Trainer(configs_forGT)

    # load RE weights
    # if configs_forGT.load_RE_weight:
    #     trainer.network.arranger.load("./arranger_taxi_STresnet_dict.pth")
    # if configs_forGT.RE_trainable is False:
    #     trainer.network.arranger.requires_grad_(False)

    # load weight
    # print('----------- load weight ------------')
    # chk = torch.load('checkpoint.chk')
    # trainer.network.load_state_dict(chk['net'])

    trainer.save_configs('config_train.pkl')
    trainer.train(dataset_train, dataset_eval, 'checkpoint.chk', datasets)
    print('\n----- training finished -----\n')

    del dataset_train, dataset_eval, trains, develops

    # test
    dataset_test = tests

    print('loading test dataloader')
    dataloader_test = DataLoader(dataset_test, batch_size=configs_forGT.batch_size_test, shuffle=False)
    chk = torch.load('checkpoint.chk')
    trainer.network.load_state_dict(chk['net'])
    print('testing...')
    begin_time = time.time()
    loss_sst_test, loss_nino_test, sc_test, er, nmae = trainer.infer(dataset=dataset_test, dataloader=dataloader_test, datasets=datasets)
    rmse = sc_test
    print(f'ER_eval=\t{er}\tNMAE=\t{nmae}\tRMSE=\t{rmse}\tTime=\t{(time.time() - begin_time)}\t'
          f'batch_training_loss: {loss_sst_test}, {loss_nino_test}')


if __name__ == '__main__':
    # train_for_neno()
    # train_for_taxibj()  # no use MinMaxScaler
    # train_for_abilene()  # no use MinMaxScaler
    train_for_geant()  # use MinMaxScaler
