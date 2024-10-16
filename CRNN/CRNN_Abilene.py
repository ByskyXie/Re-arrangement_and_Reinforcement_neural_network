# -*- coding: utf-8 -*-
import pandas as pd
import sys
import numpy as np
import keras.layers
import keras
from keras.layers import Lambda
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard, Callback
import time
import xml.etree.ElementTree as ET
import h5py
import random
from sklearn import preprocessing
import tensorflow as tf
# import tensorflow.contrib.rnn as rnn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
#hyper-parameter
#source = 5
#destination = 9
#pos = source*12+destination

# tf.device('/device:GPU:0')

# #load data  preversion
# k = 100
# TM = []
# for i in range(1,25):
#     num = k+i
#     with open('X'+str(num)[1:3],'r') as f:
#         for line in f:
#             TM.append(list(line.split(' ')))
#
# TM = np.array(TM)  # shape=(xxx,721)
# TM_real = np.zeros((48384,144))
# for i in range(48384):
#     for j in range(144):
#         TM_real[i][j] = TM[i][j*5+1]

# load
import os

random_seed = 777
random.seed(random_seed )  # set random seed for python
np.random.seed(random_seed )  # set random seed for numpy
tf.set_random_seed(random_seed )  # set random seed for tensorflow-cpu
os.environ['TF_DETERMINISTIC_OPS'] = '1' # set random seed for tensorflow-gpu

def get_traffic_matrix_abilene(path: str = '.', all_batch_size=None):
    files = os.listdir(path)  # list current path files
    # filter other file
    index = len(files) - 1
    while index >= 0:
        if files[index].find('tm.2004') == -1:
            del (files[index])
        index -= 1

    # sort file
    files.sort()  

    if all_batch_size is None:
        all_batch_size = len(files)
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
    return tms

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
    return tms

def get_traffic_matrix_taxibj(path: str = '.', all_batch_size=1000, InData=True):
    print('Begin load TaxiBj')
    f = h5py.File(path, 'r')
    print(f'Keys contains data&date, use InData:{InData}')
    tms = f['data'][:, 0, :, :] if InData else f['data'][:, 1, :, :]
    print('TaxiBj loaded')

    assert len(tms) >= all_batch_size
    tms = tms[:all_batch_size]
    return tms


# # TODO:Abilene
# datasets = get_traffic_matrix_abilene('Abilene_dataset_path', 4000)
# shape_origin = datasets.shape
# TM_real = datasets.reshape(shape_origin[0], -1)
# NUM_NODE = 12
# NUM_FLOWs = 144

# TODO:GEANT
datasets = get_traffic_matrix_geant('GEANT_dataset_path', 4000)
shape_origin = datasets.shape
TM_real = datasets.reshape(shape_origin[0], -1)
NUM_NODE = 23
NUM_FLOWs = 529

# TODO:Taxibj
# datasets = get_traffic_matrix_taxibj('TaxiBJ_dataset_path', 4000, InData=True)
# shape_origin = datasets.shape
# TM_real = datasets.reshape(shape_origin[0], -1)
# NUM_NODE = 32
# NUM_FLOWs = 1024

BATCH_SIZE = 8
HIDDEN_UNITS1=256
HIDDEN_UNITS2=512
HIDDEN_UNITS=NUM_FLOWs
LEARNING_RATE = 0.00001
EPOCH=200
VECTOR_SIZE = NUM_FLOWs*2  #the length of inter-flow correlational feature vector, the optimal length
IS_REARRANGE = False

#normalization
MAX_VALUE = TM_real.max(axis=0)
data = TM_real / TM_real.max(axis=0)
sequence_length = 10 
result = []
for index in range(data.shape[0] - sequence_length - 1):
    result.append(data[index: index + sequence_length + 1])

result = np.array(result)
# print(result.shape)  # (48373, 11, 144) construct dataset

#pos = source*12+destination
y = result[:,-1,:]
x = result[:,:-1,:]
print(y.shape)
print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.80, random_state=33)

TRAIN_EXAMPLES=x_train.shape[0]
TEST_EXAMPLES=x_test.shape[0]


class Arranger(tf.keras.layers.Layer):

    __constants__ = ['matrix_row', 'matrix_column']
    matrix_row: int
    matrix_column: int
    weight_row: tf.Tensor
    weight_column: tf.Tensor

    def __init__(self, matrix_row, matrix_column):
        super(Arranger, self).__init__()
        self.matrix_row =matrix_row
        self.matrix_column = matrix_column

        eye_row, eye_column = tf.eye(matrix_row), tf.eye(matrix_column)
        with tf.Session() as sess:
            eye_row = eye_row.eval()
            eye_column = eye_column.eval()

        row_init = tf.constant_initializer(eye_row)
        self.weight_row = self.add_weight(shape=(matrix_row, matrix_row), initializer=row_init, name='Arranger_weight_row')

        column_init = tf.constant_initializer(eye_column)
        self.weight_column = self.add_weight(shape=(matrix_column, matrix_column), initializer=column_init, name='Arranger_weight_column')

    def call(self, x, **kwargs):
        # print('x shape ', x.shape)
        out = tf.matmul(x, self.weight_column)  # rearrange column
        out = tf.matmul(self.weight_row, out)  # rearrange row
        # print('out shape ', out.shape)
        return out

    def get_inverse_matrix(self):
        inverse_column = tf.matrix_inverse(self.weight_column)
        inverse_row = tf.matrix_inverse(self.weight_row)
        return [inverse_column, inverse_row]


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




###############Convolutional Recurrent Neural Network###############
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1 padding='SAME'
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, sequence_length, NUM_FLOWs])
ys = tf.placeholder(tf.float32, [None, NUM_FLOWs])

#------------------------------------construct CNN------------------------------------------#
keep_prob = tf.placeholder(tf.float32)#dropout
#source_index = tf.placeholder(tf.float32)
#destination_index = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, NUM_NODE, NUM_NODE, 1])#144-->12*12

# TODO:Arranger [
x_image = tf.reshape(x_image, [-1, NUM_NODE, NUM_NODE])
arranger = Arranger(NUM_NODE, NUM_NODE)
x_image = arranger(x_image)
x_image = tf.reshape(x_image, [-1, NUM_NODE, NUM_NODE, 1])


## conv1 layer
print('x_image:', x_image.shape)
W_conv1 = weight_variable([3,3, 1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
#h_pool1 = max_pool_2x2(h_conv1)    
print("h_conv1:",h_conv1.shape)
## conv2 layer 
W_conv2 = weight_variable([3,3, 32, 64]) 
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) 
print("h_conv2:",h_conv2.shape)

## conv2 layer 
#W_conv3 = weight_variable([3,3, 64, 128])
#b_conv3 = bias_variable([128])
#h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3) 
#print("h_conv3:",h_conv3.shape)


## fc1 layer ##  full connection 
W_fc1 = weight_variable([NUM_FLOWs*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_conv2, [-1, NUM_FLOWs*64])
print('h_pool2_flat:', h_pool2_flat.shape)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print('h_fc1:', h_fc1.shape)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#dropout
print('h_fc1_drop:', h_fc1_drop.shape)
## fc2 layer ## full connection
W_fc2 = weight_variable([1024, VECTOR_SIZE])
b_fc2 = bias_variable([VECTOR_SIZE])
#output of CNN
cnn_out =  tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print('cnn_out pre:', cnn_out.shape)
cnn_out = tf.concat([cnn_out,tf.reshape(xs, [-1, NUM_FLOWs])],axis = 1)
print("cnn_out:",cnn_out.shape)


#------------------------------------construct LSTM------------------------------------------#
#lstm instance
lstm_cell1 = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=HIDDEN_UNITS1)
lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell1, output_keep_prob=keep_prob)
lstm_cell2 = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cel2', num_units=HIDDEN_UNITS2)
lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell2, output_keep_prob=keep_prob)
lstm_cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=HIDDEN_UNITS)
multi_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell1,lstm_cell2,lstm_cell])
#initialize to zero
init_state=multi_lstm.zero_state(batch_size=BATCH_SIZE,dtype=tf.float32)
cnn_out = tf.reshape(cnn_out,[-1, sequence_length, VECTOR_SIZE+NUM_FLOWs])
print("cnn_out:",cnn_out.shape)
#dynamic rnn  shape=(64, 10, 144)
outputs,states = tf.nn.dynamic_rnn(cell=multi_lstm,inputs=cnn_out,initial_state=init_state,dtype=tf.float32)
# rnn = keras.layers.RNN(cell=multi_lstm)
# outputs = rnn(cnn_out, initial_state=init_state)

# TODO:Arranger ]
outputs = tf.reshape(outputs[:,-1,:], [-1, NUM_NODE, NUM_NODE])
def restore_layer(inputs):
    inverse_column, inverse_row = arranger.get_inverse_matrix()
    out = tf.matmul(inputs, inverse_column)
    return tf.matmul(inverse_row, out)
outputs = Lambda(restore_layer)(outputs)
outputs = tf.reshape(outputs, [-1, 1, NUM_FLOWs])

print("outputs:",outputs.shape)
prediction = outputs[:,-1,:]
print("prediction:",prediction.shape)


#prediction = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#---------------------------------define loss and optimizer----------------------------------#
mse=tf.losses.mean_squared_error(labels=ys,predictions=prediction)
#print(loss.shape)
optimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=mse)



sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
sess.run(tf.global_variables_initializer())
results_train = []
results_test = []

for epoch in range(1,EPOCH+1):
    BEGIN_TIME = time.time()
    results = np.zeros(shape=(TEST_EXAMPLES, 1))
    train_losses=[]
    test_losses=[]
    # print("epoch:",epoch)

    if IS_REARRANGE:
        Trainable_epochs = 80
        if epoch < Trainable_epochs:
            arranger.trainable = False
        else:
            arranger.trainable = True
    else:
        arranger.trainable = False

    for j in range(TRAIN_EXAMPLES//BATCH_SIZE):
        _,train_loss=sess.run(
                fetches=(optimizer,mse),
                feed_dict={
                        xs:x_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                        ys:y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE], keep_prob: 0.2
                    }
        )
        train_losses.append(train_loss)
    # print("average training loss:", sum(train_losses) / len(train_losses))
    results_train.append(sum(train_losses) / len(train_losses))

    aer, anmae, armse = np.array(0), np.array(0), np.array(0)
    for j in range(TEST_EXAMPLES//BATCH_SIZE):
        pred,test_loss=sess.run(
                fetches=(prediction,mse),
                feed_dict={
                        xs:x_test[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                        ys:y_test[j*BATCH_SIZE:(j+1)*BATCH_SIZE], keep_prob: 1.0
                    }
        )
        test_losses.append(test_loss)
        pred = pred.copy()
        Y_train = y_test[j*BATCH_SIZE:(j+1)*BATCH_SIZE].copy()
        pred = pred * MAX_VALUE
        Y_train = Y_train * MAX_VALUE
        er, nmae, rmse = error_rate(Y_train, pred), NMAE(Y_train, pred), RMSE(Y_train, pred)
        aer, anmae, armse = aer+er, anmae+nmae, armse+rmse
    # print("average test loss:", sum(test_losses) / len(test_losses))
    print(f'Epoch {epoch}/{EPOCH}:TestLoss={sum(test_losses) / len(test_losses)} ER=\t{aer/(TEST_EXAMPLES//BATCH_SIZE)}\tNMAE=\t{anmae/(TEST_EXAMPLES//BATCH_SIZE)}'
          f'\tRMSE=\t{armse/(TEST_EXAMPLES//BATCH_SIZE)}\ttime=\t{time.time()-BEGIN_TIME}')
    results_test.append(sum(test_losses) / len(test_losses))

print("max test mse: ",max(results_test))
print("mean test mse: ",sum(results_test)/len(results_test))
print("min test mse: ",min(results_test))

data = pd.DataFrame(results_test)
data.to_csv("mse.csv")
import matplotlib.pyplot as plt

plt.plot(results_train, label='train mse')
plt.plot(results_test, label='test mse')
plt.title('CRNN')
plt.legend()
plt.show()
