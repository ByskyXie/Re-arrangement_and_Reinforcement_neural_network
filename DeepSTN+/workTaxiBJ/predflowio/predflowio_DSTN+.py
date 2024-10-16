import datetime
import sys
import shutil
import time

from keras.models import load_model
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard, Callback

from DeepSTN_net import DeepSTN
from load_data_DSTN import load_data, MatrixPredDataset, get_traffic_matrix_abilene, get_traffic_matrix_geant
from Param_DSTN_flow import *


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_not_zero_position(inputs):
    return np.clip(np.clip(np.absolute(inputs), 0, 1e-32) * 1e36, 0, 1)

def NMAE(source, target):
    """ should mul mask previous """
    source = source.reshape(-1)
    target = target.reshape(-1)
    molecular = np.absolute(source - target)
    denominator = np.absolute(target)
    not_zero_pos = get_not_zero_position(target)
    return np.sum(not_zero_pos * molecular) / np.sum(denominator)

def RMSE(source, target):
    """ should mul mask previous """
    source = source.reshape(-1)
    target = target.reshape(-1)
    not_zero_pos = get_not_zero_position(target)

    return np.sqrt(np.power(not_zero_pos*(source-target), 2).mean())

def error_rate(source, target):
    """ should mul mask previous """
    source = source.reshape(-1)
    target = target.reshape(-1)
    molecular = np.power(source - target, 2)
    denominator = np.power(target, 2)
    not_zero_pos = get_not_zero_position(target)
    return np.power(np.sum(not_zero_pos * molecular) / np.sum(denominator), 1 / 2)


class ArrangerController(Callback):
    Trainable_epochs = 120

    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.Trainable_epochs:
            self.model.get_layer('arranger_1').trainable = False
        else:
            self.model.get_layer('arranger_1').trainable = True
        return super().on_epoch_begin(epoch, logs)


class OutputObserver(Callback):
    """"
    callback to observe the output of the network
    """

    def __init__(self, X_train, Y_train, datasets=None):
        super().__init__()
        self.pred = None
        self.X_train = X_train
        self.Y_train = Y_train
        self.begin_time = None
        self.datasets = datasets

    def on_epoch_begin(self, epoch, logs=None):
        self.begin_time = time.time()
        return super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.X_train, batch_size=BATCHSIZE).copy()
        Y_train = self.Y_train.copy()

        if self.datasets is not None and self.datasets.scaler is not None:
            shape1 = pred.shape
            shape2 = Y_train.shape
            pred = datasets.scaler.inverse_transform(pred.reshape(-1, HEIGHT*WIDTH)).reshape(shape1)
            Y_train = datasets.scaler.inverse_transform(Y_train.reshape(-1, HEIGHT*WIDTH)).reshape(shape2)
        elif dataset_name == 'TaxiBJ':
            pred = pred * MAX_FLOWIO
            Y_train = Y_train * MAX_FLOWIO

        er, nmae, rmse = error_rate(Y_train, pred), NMAE(Y_train, pred), RMSE(Y_train, pred)
        print(f'Epoch {epoch}/{EPOCH}: ER=\t{er}\tNMAE=\t{nmae}\tRMSE=\t{rmse}\ttime=\t{time.time()-self.begin_time}')
        return super().on_epoch_end(epoch, logs)


def train_model(X_train, Y_train, datasets = None):
    csv_logger = CSVLogger(PATH + '/' + MODELNAME + '.log')
    checkpointer_path = PATH + '/' + MODELNAME + '.h5'
    outObser = OutputObserver(X_train, Y_train, datasets)
    checkpointer = ModelCheckpoint(filepath=checkpointer_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    LearnRate = LearningRateScheduler(lambda epoch: LR)

    model = DeepSTN(H=HEIGHT, W=WIDTH, channel=CHANNEL,
                    c=len_closeness, p=len_period, t=len_trend,
                    pre_F=pre_F, conv_F=conv_F, R_N=R_N,
                    is_plus=is_plus, plus=plus, rate=rate,
                    is_pt=is_pt, T=DAYTIMESTEP, drop=drop)

    arraContr = ArrangerController(model)
    model.fit(X_train, Y_train, epochs=EPOCH, batch_size=BATCHSIZE,
              validation_split=SPLIT, shuffle=True, verbose=False,
              callbacks=[csv_logger, LearnRate, early_stopping, outObser, arraContr])

    keras_score = model.evaluate(X_train, Y_train, verbose=1)
    rescaled_MSE = keras_score * MAX_FLOWIO * MAX_FLOWIO

    f = open(PATH + '/' + MODELNAME + '_prediction_scores.txt', 'a')
    f.write("Keras MSE on trainData, %f\n" % keras_score)
    f.write("Rescaled MSE on trainData, %f\n" % rescaled_MSE)
    f.close()

    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescaled_MSE)
    print('Model Training Ended ...', time.ctime())

    return model


def test_model(X_test, Y_test, model, datasets = None):
    print('Model Evaluation Started ...', time.ctime())

    # assert os.path.exists(PATH + '/' + MODELNAME + '.h5'), 'model is not existing'
    # model = load_model(PATH + '/' + MODELNAME + '.h5')
    model.summary()

    pred = model.predict(X_test, batch_size=BATCHSIZE)
    if datasets is not None and datasets.scaler is not None:
        pred_ = datasets.scaler.inverse_transform(pred.reshape(-1, HEIGHT*WIDTH))
        Y_test_ = datasets.scaler.inverse_transform(Y_test.reshape(-1, HEIGHT*WIDTH))
    elif dataset_name == 'TaxiBJ':
        pred_ = pred * MAX_FLOWIO
        Y_test_ = Y_test * MAX_FLOWIO
    er, nmae, rmse = error_rate(Y_test_,pred_), NMAE(Y_test_,pred_), RMSE(Y_test_,pred_)
    print(f'Test set: ER=\t{er}\tNMAE=\t{nmae}\tRMSE=\t{rmse}')

    keras_score = model.evaluate(X_test, Y_test, verbose=1, batch_size=BATCHSIZE)
    rescale_MSE = keras_score * MAX_FLOWIO * MAX_FLOWIO

    f = open(PATH + '/' + MODELNAME + '_prediction_scores.txt', 'a')
    f.write("Keras MSE on testData, %f\n" % keras_score)
    f.write("Rescaled MSE on testData, %f\n" % rescale_MSE)
    f.close()

    print('*' * 40)
    print('keras MSE', keras_score)
    print('rescaled MSE', rescale_MSE)
    print('Model Evaluation Ended ...', time.ctime())

################# Path Setting #######################
MODELNAME = 'DeepSTN+'
KEYWORD = 'predflowio_' + MODELNAME + '_' + datetime.datetime.now().strftime("%y%m%d%H%M")
PATH = '../' + KEYWORD
###########################Reproducible#############################
import numpy as np
import random
from keras import backend as K
import os
import tensorflow as tf

np.random.seed(100)
random.seed(100)
os.environ['PYTHONHASHSEED'] = '0'  # necessary for py3

tf.compat.v1.set_random_seed(100)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True
session_conf.gpu_options.visible_device_list = '0'
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)
###################################################################

if __name__ == '__main__':
    mkdir(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('DeepSTN_net.py', PATH)
    shutil.copy2('load_data_DSTN.py', PATH)
    shutil.copy2('Param_DSTN_flow.py', PATH)
    StartTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    print('#' * 50)
    print('start running at {}'.format(StartTime))
    print('model name: {}'.format(MODELNAME))
    print('#' * 50, '\n')

    X_train, Y_train, X_test, Y_test = None, None, None, None
    datasets = None

    if dataset_name == 'TaxiBJ':
        X_train, Y_train, X_test, Y_test = load_data()
    elif dataset_name == 'Abilene':
        datasets = MatrixPredDataset(path='Abilene_dataset_path', all_batch_num=4000,
                                     predict_matrix_num=1,
                                     input_matrix_num=len_closeness,
                                     fn_get_traffic_matrix=get_traffic_matrix_abilene, gpu_mode=True,
                                     sampling_rate1=1, sampling_rate2=1, len_period=len_period, len_trend=len_trend
                                     , period_interval=T_period, trend_interval=T_trend)
        X_train, Y_train, X_test, Y_test = datasets.load_pred_data(train_ratio=trainRatio)
    elif dataset_name == 'GEANT':
        datasets = MatrixPredDataset(path='GEANT_dataset_path', all_batch_num=4000,
                                     predict_matrix_num=1,
                                     input_matrix_num=len_closeness,
                                     fn_get_traffic_matrix=get_traffic_matrix_geant, gpu_mode=True,
                                     sampling_rate1=1, sampling_rate2=1, len_period=len_period, len_trend=len_trend
                                     , period_interval=T_period, trend_interval=T_trend)
        X_train, Y_train, X_test, Y_test = datasets.load_pred_data(train_ratio=trainRatio)

    model = train_model(X_train, Y_train, datasets)
    test_model(X_test, Y_test, model, datasets)
