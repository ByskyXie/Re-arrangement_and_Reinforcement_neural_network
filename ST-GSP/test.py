import time

import torch
from torch.utils.data import DataLoader
from models.stgsp import STGSP
from data.dataset import DatasetFactory
import numpy as np
from data.dataset import MatrixPredDataset
from data.dataset import get_traffic_matrix_abilene, get_traffic_matrix_geant, get_traffic_matrix_video
from torch.utils.data import random_split

maxEpoch = 340
seed = 777
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DataConfigurationFirework:
    # train
    epochs = 100
    batch_size = 32
    learning_rate = 0.000009

    # Data
    name = 'Firework'
    portion = 1.  # portion of data

    len_close = 6
    len_period = 1
    len_trend = 1
    pad_forward_period = 0
    pad_back_period = 0
    pad_forward_trend = 0
    pad_back_trend = 0

    len_all_close = len_close * 1
    len_all_period = len_period * (1 + pad_back_period + pad_forward_period)
    len_all_trend = len_trend * (1 + pad_back_trend + pad_forward_trend)
    predict_matrix_num = 1

    len_seq = len_all_close + len_all_period + len_all_trend
    cpt = [len_all_close, len_all_period, len_all_trend]

    interval_period = len_close+2
    interval_trend = len_close+1

    ext_flag = False
    timeenc_flag = 'w'  # 'm', 'w', 'd'
    rm_incomplete_flag = True
    fourty_eight = True
    previous_meteorol = True

    ext_dim = 1 # 77
    dim_flow = 1
    dim_h = 270
    dim_w = 480


class DataConfigurationGEANT:
    # train
    epochs = 100
    batch_size = 32
    learning_rate = 0.000009

    # Data
    name = 'GEANT'
    portion = 1.  # portion of data

    len_close = 6
    len_period = 1
    len_trend = 1
    pad_forward_period = 0
    pad_back_period = 0
    pad_forward_trend = 0
    pad_back_trend = 0

    len_all_close = len_close * 1
    len_all_period = len_period * (1 + pad_back_period + pad_forward_period)
    len_all_trend = len_trend * (1 + pad_back_trend + pad_forward_trend)
    predict_matrix_num = 1

    len_seq = len_all_close + len_all_period + len_all_trend
    cpt = [len_all_close, len_all_period, len_all_trend]

    interval_period = 96
    interval_trend = 24

    ext_flag = False
    timeenc_flag = 'w'  # 'm', 'w', 'd'
    rm_incomplete_flag = True
    fourty_eight = True
    previous_meteorol = True

    ext_dim = 1 # 77
    dim_flow = 1
    dim_h = 23
    dim_w = 23


class DataConfigurationAbilene:
    # train
    epochs = 100
    batch_size = 32
    learning_rate = 0.000009

    # Data
    name = 'Abilene'
    portion = 1.  # portion of data

    len_close = 6
    len_period = 1
    len_trend = 1
    pad_forward_period = 0
    pad_back_period = 0
    pad_forward_trend = 0
    pad_back_trend = 0

    len_all_close = len_close * 1
    len_all_period = len_period * (1 + pad_back_period + pad_forward_period)
    len_all_trend = len_trend * (1 + pad_back_trend + pad_forward_trend)
    predict_matrix_num = 1

    len_seq = len_all_close + len_all_period + len_all_trend
    cpt = [len_all_close, len_all_period, len_all_trend]

    interval_period = 12
    interval_trend = len_close + 1

    ext_flag = False
    timeenc_flag = 'w'  # 'm', 'w', 'd'
    rm_incomplete_flag = True
    fourty_eight = True
    previous_meteorol = True

    ext_dim = 1 # 77
    dim_flow = 1
    dim_h = 12
    dim_w = 12


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
    timeenc_flag = 'w'  # 'm', 'w', 'd'
    rm_incomplete_flag = True
    fourty_eight = True
    previous_meteorol = True

    ext_dim = 77 # 77
    dim_flow = 2
    dim_h = 32
    dim_w = 32

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(dataset='TaxiBJ'):
    set_seed(seed)
    datasets = None
    dconf, ds_factory, train_ds = None, None, None
    if dataset == 'TaxiBJ':
        dconf = DataConfiguration()
        ds_factory = DatasetFactory(dconf)
        select_pre = 0
        train_ds = ds_factory.get_test_dataset(select_pre)
    elif dataset == 'Abilene':
        dconf = DataConfigurationAbilene()
        datasets = MatrixPredDataset(path='Abilene_dataset_path', all_batch_num=4000,
                                     predict_matrix_num=dconf.predict_matrix_num,
                                     input_matrix_num=dconf.len_close,
                                     fn_get_traffic_matrix=get_traffic_matrix_abilene, gpu_mode=True,
                                     sampling_rate1=1, sampling_rate2=1, len_period=dconf.len_period,
                                     len_trend=dconf.len_trend,
                                     period_interval=dconf.interval_period,
                                     trend_interval=dconf.interval_trend)
        # scaler = datasets.scaler
        train_ds, valid_dataset, tests = random_split(
            datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)), round(0.1 * len(datasets))],
            generator=torch.Generator().manual_seed(23))
        del valid_dataset, tests
    elif dataset == 'GEANT':
        dconf = DataConfigurationGEANT()
        datasets = MatrixPredDataset(path='GEANT_dataset_path', all_batch_num=4000,
                                     predict_matrix_num=dconf.predict_matrix_num,
                                     input_matrix_num=dconf.len_close,
                                     fn_get_traffic_matrix=get_traffic_matrix_geant, gpu_mode=True,
                                     sampling_rate1=1, sampling_rate2=1, len_period=dconf.len_period,
                                     len_trend=dconf.len_trend,
                                     period_interval=dconf.interval_period,
                                     trend_interval=dconf.interval_trend)
        # scaler = datasets.scaler
        train_ds, valid_dataset, tests = random_split(
            datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)), round(0.1 * len(datasets))],
            generator=torch.Generator().manual_seed(23))
        del valid_dataset, tests
    elif dataset == 'Firework':
        dconf = DataConfigurationFirework()
        datasets = MatrixPredDataset(path='C:\Python project\WeightedSwap\Video_matrix2', all_batch_num=1250,
                                     predict_matrix_num=dconf.predict_matrix_num,
                                     input_matrix_num=dconf.len_close,
                                     fn_get_traffic_matrix=get_traffic_matrix_video, gpu_mode=True,
                                     sampling_rate1=1, sampling_rate2=1, len_period=dconf.len_period,
                                     len_trend=dconf.len_trend,
                                     period_interval=dconf.interval_period,
                                     trend_interval=dconf.interval_trend)
        # scaler = datasets.scaler
        train_ds, valid_dataset, tests = random_split(
            datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)), round(0.1 * len(datasets))],
            generator=torch.Generator().manual_seed(23))
        del valid_dataset, tests

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=32,
        shuffle=False
    )

    model = STGSP(dconf)
    # model.arranger.load("checkpoints/TaxiBJ/arranger_taxi_STresnet_dict.pth")
    # model.load("checkpoints/TaxiBJ/model_finetune.pth")
    model = model.to(device)

    model.train()

    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 170, 250, 340], gamma=0.5)
    for epoch in range(maxEpoch):
        if epoch<120:  # if matrix size is too large to convergence, you can training task model in begining 40 epochs, therefore training R&E NN
            model.arranger.requires_grad_(False)
        else:
            model.arranger.requires_grad_(True)
        trues = []
        preds = []
        begin_time = time.time()
        true, pred = None, None
        for _, (X, X_ext, Y, Y_ext) in enumerate(train_loader):
            # [32, 10, 32, 32]) [32, 5, 77] [32, 2, 32, 32] [32, 77] 10=(3 1 1)*2
            X = X.to(device)
            X_ext = X_ext.to(device)
            Y = Y.to(device)
            Y_ext = Y_ext.to(device)
            outputs = model(X, X_ext, Y_ext)

            # update model
            optimizer.zero_grad()
            loss = loss_func(Y, outputs)
            loss.backward()
            optimizer.step()
            lr_sched.step()

            if dataset == 'TaxiBJ':
                true = ds_factory.ds.mmn.inverse_transform(Y.detach().cpu().numpy())
                pred = ds_factory.ds.mmn.inverse_transform(outputs.detach().cpu().numpy())
            else:
                true = Y.detach().cpu().view(-1, 1).numpy()
                pred = outputs.detach().cpu().view(-1, 1).numpy()
                true = datasets.scaler.inverse_transform(true)
                pred = datasets.scaler.inverse_transform(pred)
            trues.append(true)
            preds.append(pred)

        trues = np.concatenate(trues, 0)
        preds = np.concatenate(preds, 0)
        mae = np.mean(np.abs(preds-trues))
        trues, preds = torch.Tensor(trues), torch.Tensor(preds)
        rmse = RMSE(trues, preds)
        er = error_rate(trues, preds)
        nmae = NMAE(trues, preds)
        print(f"Epoch[{epoch}]: ER=\t{er}\tNMAE=\t{nmae}\tRMSE=\t%.5f\tMAE=\t%.5f\t"
              f"Time=\t{(time.time()-begin_time)}" % (rmse, mae))
    torch.save(model.state_dict(), "checkpoints/TaxiBJ/model_finetune.pth")

def eval(dataset='TaxiBJ'):
    set_seed(seed)
    datasets = None
    dconf, ds_factory, test_ds = None, None, None
    if dataset == 'TaxiBJ':
        dconf = DataConfiguration()
        ds_factory = DatasetFactory(dconf)
        select_pre = 0
        test_ds = ds_factory.get_test_dataset(select_pre)
    elif dataset == 'Abilene':
        dconf = DataConfigurationAbilene()
        datasets = MatrixPredDataset(path='Abilene_dataset_path', all_batch_num=4000,
                                     predict_matrix_num=dconf.predict_matrix_num,
                                     input_matrix_num=dconf.len_close,
                                     fn_get_traffic_matrix=get_traffic_matrix_abilene, gpu_mode=True,
                                     sampling_rate1=1, sampling_rate2=1, len_period=dconf.len_period,
                                     len_trend=dconf.len_trend,
                                     period_interval=dconf.interval_period,
                                     trend_interval=dconf.interval_trend)
        # scaler = datasets.scaler
        train_ds, valid_dataset, test_ds = random_split(
            datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)), round(0.1 * len(datasets))],
            generator=torch.Generator().manual_seed(23))
        del train_ds, valid_dataset
    elif dataset == 'GEANT':
        dconf = DataConfigurationGEANT()
        datasets = MatrixPredDataset(path='GEANT_dataset_path', all_batch_num=4000,
                                     predict_matrix_num=dconf.predict_matrix_num,
                                     input_matrix_num=dconf.len_close,
                                     fn_get_traffic_matrix=get_traffic_matrix_geant, gpu_mode=True,
                                     sampling_rate1=1, sampling_rate2=1, len_period=dconf.len_period,
                                     len_trend=dconf.len_trend,
                                     period_interval=dconf.interval_period,
                                     trend_interval=dconf.interval_trend)
        # scaler = datasets.scaler
        train_ds, valid_dataset, test_ds = random_split(
            datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)), round(0.1 * len(datasets))],
            generator=torch.Generator().manual_seed(23))
        del train_ds, valid_dataset
    elif dataset == 'Firework':
        dconf = DataConfigurationGEANT()
        datasets = MatrixPredDataset(path='C:\Python project\WeightedSwap\Video_matrix2', all_batch_num=4000,
                                     predict_matrix_num=dconf.predict_matrix_num,
                                     input_matrix_num=dconf.len_close,
                                     fn_get_traffic_matrix=get_traffic_matrix_video, gpu_mode=True,
                                     sampling_rate1=1, sampling_rate2=1, len_period=dconf.len_period,
                                     len_trend=dconf.len_trend,
                                     period_interval=dconf.interval_period,
                                     trend_interval=dconf.interval_trend)
        # scaler = datasets.scaler
        train_ds, valid_dataset, test_ds = random_split(
            datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)), round(0.1 * len(datasets))],
            generator=torch.Generator().manual_seed(23))
        del train_ds, valid_dataset

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=32,
        shuffle=False
    )

    model = STGSP(dconf)
    model.load("checkpoints/TaxiBJ/model_finetune.pth")
    model = model.to(device)

    model.eval()
    trues = []
    preds = []
    true, pred = None, None
    with torch.no_grad():
        for _, (X, X_ext, Y, Y_ext) in enumerate(test_loader):
            X = X.to(device)
            X_ext = X_ext.to(device) 
            Y = Y.to(device) 
            Y_ext = Y_ext.to(device)
            outputs = model(X, X_ext, Y_ext)
            if dataset == 'TaxiBJ':
                true = ds_factory.ds.mmn.inverse_transform(Y.detach().cpu().numpy())
                pred = ds_factory.ds.mmn.inverse_transform(outputs.detach().cpu().numpy())
            else:
                true = Y.detach().cpu().view(-1, 1).numpy()
                pred = outputs.detach().cpu().view(-1, 1).numpy()
                true = datasets.scaler.inverse_transform(true)
                pred = datasets.scaler.inverse_transform(pred)

            trues.append(true)
            preds.append(pred)
    trues = np.concatenate(trues, 0)
    preds = np.concatenate(preds, 0)
    mae = np.mean(np.abs(preds-trues))
    trues, preds = torch.Tensor(trues), torch.Tensor(preds)
    rmse = RMSE(trues, preds)
    er = error_rate(trues, preds)
    nmae = NMAE(trues, preds)
    print(f"TestSet: ER=\t{er}\tNMAE=\t{nmae}\tRMSE=\t%.5f\tMAE=\t%.5f\t" % (rmse, mae))



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




if __name__ == '__main__':
    print("l_c:", DataConfiguration.len_close, "l_p:", DataConfiguration.len_period, "l_t:", DataConfiguration.len_trend)
    train(dataset='TaxiBJ')
    print('-------Evaluation---------')
    eval(dataset='TaxiBJ')
    