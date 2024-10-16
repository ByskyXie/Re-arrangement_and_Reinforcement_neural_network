from model import PredictionModel, FillModel, MapInterpolationModel
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from adapter import MatrixPredDataset, MatrixFillDataset, MatrixDataset, get_traffic_matrix_geant, get_traffic_matrix_abilene\
    , get_traffic_matrix_video, get_traffic_matrix_taxibj
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import random_split

device = torch.device('cuda:0')


def losses(pred, gt):
    return F.l1_loss(pred, gt)
    # return F.mse_loss(pred, gt)


class MissedMAE(torch.nn.Module):
    """
        loss function, only compute the MAE of missing positions.
    """
    def __init__(self) -> None:
        super(MissedMAE, self).__init__()
        self.mae = torch.nn.L1Loss()

    def forward(self, source, target):
        source, target = source.reshape(-1), target.reshape(-1)
        # return torch.mean(torch.pow(torch.abs(source-target)/torch.abs(source+target), 1.2)) \
        # + torch.pow(torch.mean(target*(torch.abs(source-target)/torch.abs(source/100+target+0.0001))), 1.2)
        not_zero_pos = get_not_zero_position(target)
        s = source * not_zero_pos
        t = target
        return torch.mean(self.mae(s, t))
missmae = MissedMAE()


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

def plot_matrix(pred, gt):
    """
        visualize the prediction and ground-truth
    """
    plt.subplot(3, 2, 1)
    plt.imshow(pred[0][0].detach().numpy(), interpolation='nearest', cmap='Blues', origin='lower')
    plt.subplot(3, 2, 2)
    plt.imshow(gt[0][0].detach().numpy(), interpolation='nearest', cmap='Blues', origin='lower')

    plt.subplot(3, 2, 3)
    plt.imshow(pred[0][1].detach().numpy(), interpolation='nearest', cmap='Blues', origin='lower')
    plt.subplot(3, 2, 4)
    plt.imshow(gt[0][1].detach().numpy(), interpolation='nearest', cmap='Blues', origin='lower')

    plt.subplot(3, 2, 5)
    plt.imshow(pred[0][2].detach().numpy(), interpolation='nearest', cmap='Blues', origin='lower')
    plt.subplot(3, 2, 6)
    plt.imshow(gt[0][2].detach().numpy(), interpolation='nearest', cmap='Blues', origin='lower')
    plt.show()



def train_tim_model(input_matrix_num, predict_matrix_num, batch_size, in_channels, epochs, sampling_rate1, sampling_rate2):

    # timNetwork = torch.load('model.pth')  # load

    # # TODO: TAA dataset
    # timNetwork = MapInterpolationModel(input_matrix_num, predict_matrix_num, in_channels=in_channels, blocks_num=2, matrix_row=144, matrix_column=176).to(device)
    # datasets = MatrixDataset(path='./dataset/TAA', all_batch_num=600,
    #                       predict_matrix_num=predict_matrix_num, input_matrix_num=input_matrix_num,
    #                       fn_get_traffic_matrix=get_traffic_matrix_video, gpu_mode=True,
    #                       sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # TODO:GEANT dataset
    # timNetwork = MapInterpolationModel(input_matrix_num, predict_matrix_num, in_channels=in_channels, blocks_num=4
    #                               , matrix_row=23, matrix_column=23).to(device)
    # datasets = MatrixDataset(path='./dataset/GEANT', all_batch_num=1250,
    #                          predict_matrix_num=predict_matrix_num, input_matrix_num=input_matrix_num,
    #                          fn_get_traffic_matrix=get_traffic_matrix_geant, gpu_mode=True,
    #                          sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # TODO: abilene dataset
    timNetwork = MapInterpolationModel(input_matrix_num, predict_matrix_num, in_channels=in_channels, blocks_num=5, matrix_row=12, matrix_column=12).to(device)
    datasets = MatrixDataset(path='./dataset/Abilene', all_batch_num=1250,
                          predict_matrix_num=predict_matrix_num, input_matrix_num=input_matrix_num,
                          fn_get_traffic_matrix=get_traffic_matrix_abilene, gpu_mode=True,
                          sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)


    # TODO: TaxiBj dataset
    # timNetwork = MapInterpolationModel(input_matrix_num, predict_matrix_num, in_channels=in_channels, blocks_num=4
    #                               , matrix_row=32, matrix_column=32).to(device)
    # datasets = MatrixDataset(path='TaxiBJ_dataset_path', all_batch_num=1250,
    #                          predict_matrix_num=predict_matrix_num, input_matrix_num=input_matrix_num,
    #                          fn_get_traffic_matrix=get_traffic_matrix_taxibj, gpu_mode=True,
    #                          sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # TODO: loader dataset
    trains, develops, tests = random_split(datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)),
                                                      round(0.1 * len(datasets))],
                                           generator=torch.Generator().manual_seed(23))
    trainloader = DataLoader(trains, batch_size=batch_size, shuffle=True)
    developloader = DataLoader(develops, batch_size=batch_size, shuffle=True)
    testsloader = DataLoader(tests, batch_size=batch_size, shuffle=True)

    # define optimizer
    optimer = torch.optim.Adam(timNetwork.parameters(), lr=0.001)
    # training model
    for epoch in range(epochs):
        # train
        beg_time = time.time()
        avg_loss, avg_er, avg_nmae, avg_rmse, counter = 0, 0, 0, 0, 0

        ## if matrix size is too large to convergence, you can training task model in begining 40 epochs, therefore training R&E NN
        # if epoch<40:
        #     timNetwork.arranger.requires_grad_(False)
        # else:
        #     timNetwork.arranger.requires_grad_(True)

        for train, valid, time_seq in trainloader:
            pred = timNetwork(train, time_seq)
            # print(pred.shape, valid.shape, train.shape)
            pred = torch.squeeze(pred)
            valid = torch.squeeze(valid)
            loss = losses(pred, valid)

            optimer.zero_grad()
            loss.backward()
            optimer.step()

            # plot image
            # if error_rate(pred[0][predict_matrix_num//2], valid[0][input_matrix_num//2])>0.3:
            # if counter == 10:
            #     plot_matrix(pred, valid)

            avg_loss += loss.detach()
            avg_er += error_rate(pred, valid).detach()
            avg_nmae += NMAE(pred, valid).detach()
            avg_rmse += torch.sqrt(torch.pow(pred-valid, 2).mean()).detach()
            counter += 1
        print(f'Epoch:{epoch}\tloss=\t{avg_loss / counter}\ttrain_ER=\t{avg_er / counter}'
              f'\ttrain_NMAE=\t{avg_nmae / counter}\ttrain_RMSE=\t{avg_rmse / counter}', end='\t')

        # develop performance
        avg_loss, avg_er, avg_nmae, counter = 0, 0, 0, 0
        for develop, valid, time_seq in developloader:
            pred = timNetwork(develop, time_seq)
            pred, valid = pred.cpu(), valid.cpu()
            pred = torch.squeeze(pred)
            valid = torch.squeeze(valid)
            avg_er += error_rate(pred, valid).detach()
            avg_nmae += NMAE(pred, valid).detach()
            avg_rmse += torch.sqrt(torch.pow(pred-valid, 2).mean()).detach()
            counter += 1
        print(f'Develop_ER=\t{avg_er / counter}\tDevelop_NMAE=\t{avg_nmae / counter}\tDevelop_RMSE=\t{avg_rmse / counter}\tTime:{time.time()-beg_time}')
        if epoch%5==0:
            torch.save(timNetwork, 'model.pth')
    # test performance
    avg_loss, avg_er, avg_nmae, avg_rmse, counter = 0, 0, 0, 0, 0
    for test, valid, time_seq in testsloader:
        pred = timNetwork(test, time_seq)
        pred, valid = pred.cpu(), valid.cpu()
        pred = torch.squeeze(pred)
        valid = torch.squeeze(valid)
        avg_er += error_rate(pred, valid).detach()
        avg_nmae += NMAE(pred, valid).detach()
        avg_rmse += torch.sqrt(torch.pow(pred-valid, 2).mean()).detach()
        counter += 1
    print(f'Test_ER=\t{avg_er / counter}\tTest_NMAE=\t{avg_nmae / counter}\tTest_RMSE=\t{avg_rmse / counter}')
    # save model
    torch.save(timNetwork, 'model.pth')
    # save parameter of relin0, the relin0 is the weighted permutation matrix
    # torch.save(timNetwork.arranger.relin0, 'FCN_bj_relin0.pth')




def train_pred_model(input_matrix_num, predict_matrix_num, batch_size, in_channels, epochs, sampling_rate1, sampling_rate2):

    # predModel = torch.load('model.pth')  # load

    # TODO: TAA dataset
    # predModel = PredictionModel(input_matrix_num, predict_matrix_num, in_channels=in_channels, blocks_num=2, matrix_row=144, matrix_column=176).to(device)
    # datasets = MatrixPredDataset(path='./dataset/TAA', all_batch_num=1200,
    #                       predict_matrix_num=predict_matrix_num, input_matrix_num=input_matrix_num,
    #                       fn_get_traffic_matrix=get_traffic_matrix_video, gpu_mode=True,
    #                       sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # TODO: GEANT dataset
    # predModel = PredictionModel(input_matrix_num, predict_matrix_num, in_channels=in_channels, blocks_num=4
    #                               , matrix_row=23, matrix_column=23).to(device)
    # datasets = MatrixPredDataset(path='./dataset/GEANT', all_batch_num=2000,
    #                          predict_matrix_num=predict_matrix_num, input_matrix_num=input_matrix_num,
    #                          fn_get_traffic_matrix=get_traffic_matrix_geant, gpu_mode=True,
    #                          sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # TODO: abilene dataset
    predModel = PredictionModel(input_matrix_num, predict_matrix_num, in_channels=in_channels, blocks_num=5, matrix_row=12, matrix_column=12).to(device)
    datasets = MatrixPredDataset(path='./dataset/Abilene', all_batch_num=2000,
                          predict_matrix_num=predict_matrix_num, input_matrix_num=input_matrix_num,
                          fn_get_traffic_matrix=get_traffic_matrix_abilene, gpu_mode=True,
                          sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # TODO: TaxiBj dataset
    # predModel = PredictionModel(input_matrix_num, predict_matrix_num, in_channels=in_channels, blocks_num=4
    #                               , matrix_row=32, matrix_column=32).to(device)
    # datasets = MatrixPredDataset(path='TaxiBJ_dataset_path', all_batch_num=2000,
    #                          predict_matrix_num=predict_matrix_num, input_matrix_num=input_matrix_num,
    #                          fn_get_traffic_matrix=get_traffic_matrix_taxibj, gpu_mode=True,
    #                          sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # for m in predModel.parameters():
    #     print(type(m), m.numel())

    # TODO: loader dataset
    trains, develops, tests = random_split(datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)),
                                                      round(0.1 * len(datasets))],
                                           generator=torch.Generator().manual_seed(23))
    trainloader = DataLoader(trains, batch_size=batch_size, shuffle=True)
    developloader = DataLoader(develops, batch_size=batch_size, shuffle=True)
    testsloader = DataLoader(tests, batch_size=batch_size, shuffle=True)

    # TODO: portability. load weight from TIM model
    # relin0 = torch.load('FCN_video2_relin0.pth')
    # predModel.arranger.relin0 = relin0
    # predModel.arranger.requires_grad_(False)

    # define optimizer
    optimer = torch.optim.Adam(predModel.parameters(), lr=0.0005)
    # training model
    for epoch in range(epochs):
        # train
        beg_time = time.time()
        avg_loss, avg_er, avg_nmae, avg_rmse, counter = 0, 0, 0, 0, 0

        # if matrix size is too large to convergence, you can training task model in begining 40 epochs, therefore training R&E NN
        # if epoch<40:
        #     predModel.arranger.requires_grad_(False)
        # else:
        #     predModel.arranger.requires_grad_(True)

        for train, valid, time_seq in trainloader:
            pred = predModel(train, time_seq)
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
            avg_rmse += torch.sqrt(torch.pow(pred-valid, 2).mean()).detach()
            counter += 1
        print(f'Epoch:{epoch}\tloss=\t{avg_loss / counter}\ttrain_ER=\t{avg_er / counter}'
              f'\ttrain_NMAE=\t{avg_nmae / counter}\ttrain_RMSE=\t{avg_rmse / counter}', end='\t')

        # develop performance
        avg_loss, avg_er, avg_nmae, counter = 0, 0, 0, 0
        for develop, valid, time_seq in developloader:
            pred = predModel(develop, time_seq)
            pred = torch.squeeze(pred)
            valid = torch.squeeze(valid)
            pred, valid = pred.cpu(), valid.cpu()
            avg_er += error_rate(pred, valid).detach()
            avg_nmae += NMAE(pred, valid).detach()
            avg_rmse += torch.sqrt(torch.pow(pred-valid, 2).mean()).detach()
            counter += 1
        print(f'Develop_ER=\t{avg_er / counter}\tDevelop_NMAE=\t{avg_nmae / counter}\tDevelop_RMSE=\t{avg_rmse / counter}\tTime:{time.time()-beg_time}')
        if epoch%5==0:
            torch.save(predModel, 'model.pth')
    # test performance
    avg_loss, avg_er, avg_nmae, avg_rmse, counter = 0, 0, 0, 0, 0
    for test, valid, time_seq in testsloader:
        pred = predModel(test, time_seq)
        pred = torch.squeeze(pred)
        valid = torch.squeeze(valid)
        pred, valid = pred.cpu(), valid.cpu()
        avg_er += error_rate(pred, valid).detach()
        avg_nmae += NMAE(pred, valid).detach()
        avg_rmse += torch.sqrt(torch.pow(pred-valid, 2).mean()).detach()
        counter += 1
    print(f'Test_ER=\t{avg_er / counter}\tTest_NMAE=\t{avg_nmae / counter}\tTest_RMSE=\t{avg_rmse / counter}')
    # save model
    torch.save(predModel, 'model.pth')
    # save parameter of relin0, the relin0 is the weighted permutation matrix
    # torch.save(predModel.arranger.relin0, 'FCN_video_relin0.pth')


def train_fill_model(input_matrix_num, predict_matrix_num, batch_size, in_channels, epochs, sampling_rate1, sampling_rate2):

    # fillModel = torch.load('model.pth')  # load

    # TODO:TAA
    # fillModel = FillModel(input_matrix_num, in_channels=in_channels, blocks_num=2, matrix_row=144, matrix_column=176).to(device)
    # datasets = MatrixFillDataset(path='./dataset/TAA', all_batch_num=1200,
    #                       input_matrix_num=input_matrix_num,
    #                       fn_get_traffic_matrix=get_traffic_matrix_video, gpu_mode=True,
    #                       sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # TODO:GEANT
    # fillModel = FillModel(input_matrix_num, in_channels=in_channels, blocks_num=4
    #                               , matrix_row=23, matrix_column=23).to(device)
    # datasets = MatrixFillDataset(path='./dataset/GEANT', all_batch_num=2000,
    #                          input_matrix_num=input_matrix_num,
    #                          fn_get_traffic_matrix=get_traffic_matrix_geant, gpu_mode=True,
    #                          sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # TODO:abilene
    # fillModel = FillModel(input_matrix_num, in_channels=in_channels, blocks_num=5, matrix_row=12, matrix_column=12).to(device)
    # datasets = MatrixFillDataset(path='./dataset/Abilene', all_batch_num=2000,
    #                       input_matrix_num=input_matrix_num,
    #                       fn_get_traffic_matrix=get_traffic_matrix_abilene, gpu_mode=True,
    #                       sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # for m in fillModel.parameters():
    #     print(type(m), m.numel())

    # TODO:TaxiBj
    fillModel = FillModel(input_matrix_num, in_channels=in_channels, blocks_num=4
                                  , matrix_row=32, matrix_column=32).to(device)
    datasets = MatrixFillDataset(path='TaxiBJ_dataset_path', all_batch_num=2000,
                             input_matrix_num=input_matrix_num,
                             fn_get_traffic_matrix=get_traffic_matrix_taxibj, gpu_mode=True,
                             sampling_rate1=sampling_rate1, sampling_rate2=sampling_rate2)

    # TODO:loader dataset
    trains, develops, tests = random_split(datasets, [round(0.8 * len(datasets)), round(0.1 * len(datasets)),
                                                      round(0.1 * len(datasets))],
                                           generator=torch.Generator().manual_seed(23))
    trainloader = DataLoader(trains, batch_size=batch_size, shuffle=True)
    developloader = DataLoader(develops, batch_size=batch_size, shuffle=True)
    testsloader = DataLoader(tests, batch_size=batch_size, shuffle=True)

    # portability. load weight from TIM model
    # relin0 = torch.load('FCN_geant_relin0.pth')
    # fillModel.arranger.relin0 = relin0
    # fillModel.arranger.requires_grad_(False)

    # define optimizer
    optimer = torch.optim.Adam(fillModel.parameters(), lr=0.0005)
    # training model
    for epoch in range(epochs):
        # train
        beg_time = time.time()
        avg_loss, avg_er, avg_nmae, avg_rmse, counter = 0, 0, 0, 0, 0

        # if epoch<40: # if matrix size is too large to convergence, you can training task model in begining 40 epochs, therefore training R&E NN
        #     fillModel.arranger.requires_grad_(False)
        # else:
        #     fillModel.arranger.requires_grad_(True)

        for train, valid, time_seq in trainloader:
            pred = fillModel(train, time_seq)
            # print(pred.shape, valid.shape, train.shape)
            pred = torch.squeeze(pred)
            valid = torch.squeeze(valid)
            loss = missmae(pred, train)

            optimer.zero_grad()
            loss.backward()
            optimer.step()

            avg_loss += loss.detach()
            avg_er += get_predict_error_rate(pred, valid, train).detach()
            avg_nmae += get_predict_NMAE(pred, valid, train).detach()
            avg_rmse += get_predict_RMSE(pred, valid, train).detach()
            counter += 1
        print(f'Epoch:{epoch}\tloss=\t{avg_loss / counter}\ttrain_ER=\t{avg_er / counter}'
              f'\ttrain_NMAE=\t{avg_nmae / counter}\ttrain_RMSE=\t{avg_rmse / counter}', end='\t')

        # develop performance
        avg_loss, avg_er, avg_nmae, counter = 0, 0, 0, 0
        for develop, valid, time_seq in developloader:
            pred = fillModel(develop, time_seq)
            pred = torch.squeeze(pred)
            valid = torch.squeeze(valid)
            pred, valid, develop = pred.cpu(), valid.cpu(), develop.cpu()
            avg_er += get_predict_error_rate(pred, valid, develop).detach()
            avg_nmae += get_predict_NMAE(pred, valid, develop).detach()
            avg_rmse += get_predict_RMSE(pred, valid, develop).detach()
            counter += 1
        print(f'Develop_ER=\t{avg_er / counter}\tDevelop_NMAE=\t{avg_nmae / counter}\tDevelop_RMSE=\t{avg_rmse / counter}\tTime:{time.time()-beg_time}')
        if epoch%5 == 0:
            torch.save(fillModel, 'model.pth')
    # test performance
    avg_loss, avg_er, avg_nmae, avg_rmse, counter = 0, 0, 0, 0, 0
    for test, valid, time_seq in testsloader:
        pred = fillModel(test, time_seq)
        pred = torch.squeeze(pred)
        valid = torch.squeeze(valid)
        pred, valid, test = pred.cpu(), valid.cpu(), test.cpu()
        avg_er += get_predict_error_rate(pred, valid, test).detach()
        avg_nmae += get_predict_NMAE(pred, valid, test).detach()
        avg_rmse += get_predict_RMSE(pred, valid, test).detach()
        counter += 1
    print(f'Test_ER=\t{avg_er / counter}\tTest_NMAE=\t{avg_nmae / counter}\tTest_RMSE=\t{avg_rmse / counter}')
    # save model
    torch.save(fillModel, 'model.pth')
    # save parameter of relin0, the relin0 is the weighted permutation matrix
    torch.save(fillModel.arranger.relin0, 'FCN_video2_relin0.pth')



if __name__ == '__main__':
    input_matrix_num = 8
    predict_matrix_num = 1
    batch_size = 8
    in_channels = 1
    epochs = 200
    sampling_rate1 = 1  # sampling rate of matrix
    sampling_rate2 = 1  # sampling rate of time interval
    print(f'[sampling_rate1={sampling_rate1}, sampling_rate2={sampling_rate2}, epochs={epochs}]')

    # train the temporal interpolation model
    # train_tim_model(input_matrix_num, predict_matrix_num, batch_size, in_channels, epochs, sampling_rate1, sampling_rate2)

    # train the prediction model
    train_pred_model(input_matrix_num, predict_matrix_num, batch_size, in_channels, epochs, sampling_rate1, sampling_rate2)

    # train the missing completion model
    # train_fill_model(input_matrix_num, predict_matrix_num, batch_size, in_channels, epochs, sampling_rate1, sampling_rate2)



