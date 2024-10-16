INTERVAL = 30
TIMESTEP = 6
DAYTIMESTEP = int(24 * 60 / INTERVAL)
WEEKTIMESTEP = DAYTIMESTEP * 7
CITY = 'TaxiBJ'
HEIGHT = 32
WIDTH = 32
CHANNEL = 2  # todo: input channel, TaxiBJ=2, abilene=1
BATCHSIZE = 4
SPLIT = 0.2
LR = 0.0001
EPOCH = 200
LOSS = 'mse'
OPTIMIZER = 'adam'
MAX_FLOWIO = 1292.0
dataPath = '../../{}/'.format(CITY)
dataFile_lst = [dataPath + 'TaxiBJ%i.npy'%x for x in range(13,17)]
timeFile = dataPath + 'TaxiBJ_timestamps.npy'
trainRatio = 0.8

len_closeness = 3  # length of closeness dependent sequence
len_period = 1  # length of peroid dependent sequence
len_trend = 1  # length of trend dependent sequence
T_closeness, T_period, T_trend = 1, DAYTIMESTEP, DAYTIMESTEP * 7

pre_F = 32  # filter size of conv
conv_F = 32  # input channels of ResPlus
R_N = 2  # nums of ResPlus
is_plus = True
plus = 8  # channels for long range spatial dependence
rate = 4  # pooling size
is_pt = False
drop = 0.1
multi_scale_fusion = True

dataset_name = 'TaxiBJ'  # add for other datset
if dataset_name == 'Abilene':
    LR = 0.0001
    BATCHSIZE = 16
    HEIGHT = 12
    WIDTH = 12
    CHANNEL = 1
    len_closeness = 6
    len_period = 1
    len_trend = 1
    DAYTIMESTEP = int(24 * 60 / 5)
    T_period, T_trend = 12, 24
elif dataset_name == 'GEANT':
    LR = 0.00001
    BATCHSIZE = 4
    HEIGHT = 23
    WIDTH = 23
    CHANNEL = 1
    len_closeness = 6
    len_period = 1
    len_trend = 1
    DAYTIMESTEP = int(24 * 60 / 15)
    T_period, T_trend = 24, 48
