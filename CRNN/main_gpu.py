from model import RCNN
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from adapter import MatrixPredDataset, MatrixFillDataset, MatrixDataset, get_traffic_matrix_geant, get_traffic_matrix_abilene\
    , get_traffic_matrix_video, get_traffic_matrix_taxibj
import torch.nn.functional as F
# from matplotlib import pyplot as plt
from torch.utils.data import random_split
import random
import numpy as np
from early_stopping import EarlyStopping

device = torch.device('cuda:0')


def losses(pred, gt):
    return F.l1_loss(pred, gt)
    # return F.mse_loss(pred, gt)


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


def error_rate(source, target):
    """ should mul mask previous """
    source = source.reshape(-1)
    target = target.reshape(-1)
    molecular = torch.pow(source - target, 2)
    denominator = torch.pow(target, 2)
    not_zero_pos = get_not_zero_position(target)
    return torch.pow(torch.sum(not_zero_pos * molecular) / torch.sum(denominator), 1 / 2)


def get_predict_error_rate(x_fake, x_real, x_miss):
    zero_pos = (1 - get_not_zero_position(x_miss)) * get_not_zero_position(x_real)
    return error_rate(zero_pos * x_fake, zero_pos * x_real)

def get_predict_NMAE(x_fake, x_real, x_miss):
    zero_pos = (1 - get_not_zero_position(x_miss)) * get_not_zero_position(x_real)
    return NMAE(zero_pos * x_fake, zero_pos * x_real)

def get_predict_RMSE(x_fake, x_real, x_miss):
    zero_pos = (1 - get_not_zero_position(x_miss)) * get_not_zero_position(x_real)
    return torch.nn.functional.l1_loss(zero_pos * x_fake, zero_pos * x_real)




def trainRCNN():
    early_stopping = EarlyStopping(".\\")

    BATCH_SIZE = 8
    LEARNING_RATE = 0.00001
    EPOCH = epochs
    using_early_stop = False

    # TODO:GEANT
    rcnn = RCNN(input_matrix_num, predict_matrix_num, in_channels=in_channels, matrix_row=23, matrix_column=23).to(device)
    datasets = MatrixPredDataset(path='GEANT_dataset_path', all_batch_num=4000,
                             predict_matrix_num=predict_matrix_num, input_matrix_num=input_matrix_num,
                             fn_get_traffic_matrix=get_traffic_matrix_geant, gpu_mode=True,
                             sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # # TODO:abilene
    # rcnn = RCNN(input_matrix_num, predict_matrix_num, in_channels=in_channels, matrix_row=12, matrix_column=12).to(device)
    # datasets = MatrixPredDataset(path='Abilene_dataset_path', all_batch_num=4000,
    #                       predict_matrix_num=predict_matrix_num, input_matrix_num=input_matrix_num,
    #                       fn_get_traffic_matrix=get_traffic_matrix_abilene, gpu_mode=True,
    #                       sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # # TODO:TaxiBj
    # rcnn = RCNN(input_matrix_num, predict_matrix_num, in_channels=in_channels, matrix_row=32, matrix_column=32).to(device)
    # datasets = MatrixPredDataset(path='TaxiBJ_dataset_path', all_batch_num=4000,
    #                          predict_matrix_num=predict_matrix_num, input_matrix_num=input_matrix_num,
    #                          fn_get_traffic_matrix=get_traffic_matrix_taxibj, gpu_mode=True,
    #                          sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    MAX_V, MIN_V = datasets.tms.max(), datasets.tms.min()

    trains, develops, tests = random_split(datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)),
                                                      round(0.1 * len(datasets))],
                                           generator=torch.Generator().manual_seed(23))
    trainloader = DataLoader(trains, batch_size=BATCH_SIZE, shuffle=True)
    developloader = DataLoader(develops, batch_size=BATCH_SIZE, shuffle=True)
    testsloader = DataLoader(tests, batch_size=BATCH_SIZE, shuffle=True)

    # define optimizer
    optimer = torch.optim.Adam(rcnn.parameters(), lr=LEARNING_RATE)
    # training model
    for epoch in range(EPOCH):
        # train
        beg_time = time.time()
        avg_loss, avg_er, avg_nmae, avg_rmse, counter = 0, 0, 0, 0, 0
        if epoch<80:  # for stability
            rcnn.arranger.requires_grad_(False)
        else:
            rcnn.arranger.requires_grad_(True)
        for train, valid, time_seq in trainloader:
            pred = rcnn(train)
            pred = pred*MAX_V
            # print(pred.shape, valid.shape, train.shape)
            pred = torch.squeeze(pred)
            valid = torch.squeeze(valid)
            loss = losses(pred, valid)

            optimer.zero_grad()
            loss.backward()
            optimer.step()

            avg_loss += loss.detach()
            avg_er += error_rate(pred, valid).detach()
            avg_nmae += NMAE(pred, valid).detach()
            avg_rmse += torch.sqrt(torch.pow(pred - valid, 2).mean()).detach()
            counter += 1
        print(f'Epoch:{epoch}\tloss=\t{avg_loss / counter}\ttrain_ER=\t{avg_er / counter}'
              f'\ttrain_NMAE=\t{avg_nmae / counter}\ttrain_RMSE=\t{avg_rmse / counter}', end='\t')

        # develop
        avg_loss, avg_er, avg_nmae, counter = 0, 0, 0, 0
        for develop, valid, time_seq in developloader:
            pred = rcnn(develop)
            pred = pred*MAX_V
            pred = torch.squeeze(pred)
            valid = torch.squeeze(valid)
            pred, valid = pred.cpu(), valid.cpu()
            avg_er += error_rate(pred, valid).detach()
            avg_nmae += NMAE(pred, valid).detach()
            avg_rmse += torch.sqrt(torch.pow(pred - valid, 2).mean()).detach()
            counter += 1
        print(f'Develop_ER=\t{avg_er / counter}\tDevelop_NMAE=\t{avg_nmae / counter}\tDevelop_RMSE=\t{avg_rmse / counter}\tTime:{time.time() - beg_time}')
        early_stopping(avg_er, rcnn)
        if epoch % 5 == 0:
            torch.save(rcnn, 'rcnn.pth')

        # Early stopping setting
        if using_early_stop and early_stopping.early_stop:
            print("--------------- Early stopping ------------------")
            rcnn.load_state_dict(torch.load('best_network_early_stopping.pth'))
            break  
    # test performance
    avg_loss, avg_er, avg_nmae, avg_rmse, counter = 0, 0, 0, 0, 0
    for test, valid, time_seq in testsloader:
        pred = rcnn(test)
        pred = pred*MAX_V
        pred = torch.squeeze(pred)
        valid = torch.squeeze(valid)
        pred, valid = pred.cpu(), valid.cpu()
        avg_er += error_rate(pred, valid).detach()
        avg_nmae += NMAE(pred, valid).detach()
        avg_rmse += torch.sqrt(torch.pow(pred - valid, 2).mean()).detach()
        counter += 1
    print(f'Test_ER=\t{avg_er / counter}\tTest_NMAE=\t{avg_nmae / counter}\tTest_RMSE=\t{avg_rmse / counter}')
    # save model
    # torch.save(rcnn, 'rcnn.pth')


def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    set_seed(777)

    input_matrix_num = 8
    predict_matrix_num = 1
    batch_size = 8
    in_channels = 1
    epochs = 200
    sampling_rate1 = 1  # sampling rate of matrix
    sampling_rate2 = 1  # sampling rate of time interval
    print(f'epochs={epochs}]')

    trainRCNN()



