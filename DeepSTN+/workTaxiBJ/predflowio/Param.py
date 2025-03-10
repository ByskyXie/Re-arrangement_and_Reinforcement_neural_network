INTERVAL = 30
TIMESTEP = 6
DAYTIMESTEP = int(24 * 60 / INTERVAL)
WEEKTIMESTEP = DAYTIMESTEP * 7
CITY = 'TaxiBJ'
HEIGHT = 32
WIDTH = 32
CHANNEL = 2
BATCHSIZE = 16
SPLIT = 0.2
LEARN = 0.0001
EPOCH = 200
LOSS = 'mse'
OPTIMIZER = 'adam'
MAX_FLOWIO = 1292.0
dataPath = '../../{}/'.format(CITY)
dataFile_lst = [dataPath + 'TaxiBJ%i.npy'%x for x in range(13,17)]
timeFile = dataPath + 'TaxiBJ_timestamps.npy'
trainRatio = 0.8