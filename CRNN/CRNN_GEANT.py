# -*- coding: utf-8 -*-
import xml.dom.minidom as xmldom
import os 
import numpy as np
import pandas as pd

TMs = []
file_names = os.listdir(r'/home/gaokaihui/TM_prediction/GEANT/data/')
file_names = sorted(file_names)
for info in file_names:# os.listdir(r'/home/gaokaihui/TM_prediction/GEANT/GEANT/data/traffic-matrices/'):
    tm = np.zeros((23,23))
    domain = os.path.abspath(r'/home/gaokaihui/TM_prediction/GEANT/data/') #获取文件夹的路径，此处其实没必要这么写，目的是为了熟悉os的文件夹操作
    info = os.path.join(domain,info) #将路径与文件名结合起来就是每个文件的完整路径
    domobj = xmldom.parse(info)
    #print("xmldom.parse:", type(domobj))
    elementobj = domobj.documentElement
    #print ("domobj.documentElement:", type(elementobj))
    
    #获得子标签
    subElementObj = elementobj.getElementsByTagName("src")
    #print ("getElementsByTagName:", type(subElementObj))
    #print (len(subElementObj))
    
    for i in range(len(subElementObj)):
        #print (subElementObj[i].getAttribute("id"))
        sub_subElementObj = subElementObj[i].getElementsByTagName("dst")
        #print (len(sub_subElementObj))
        for j in range(len(sub_subElementObj)):
            #print (sub_subElementObj[j].getAttribute("id"))
            #print (sub_subElementObj[j].firstChild.data)
            tm[int(subElementObj[i].getAttribute("id"))-1][int(sub_subElementObj[j].getAttribute("id"))-1] = float(sub_subElementObj[j].firstChild.data)
    #print(tm)
    tm = tm.reshape(23*23)
    TMs.append(tm)
    
TMs = np.array(TMs)
print(TMs.shape)
NUM_NODE = 23
NUM_FLOWs = 529
from sklearn.model_selection import train_test_split


BATCH_SIZE=32
HIDDEN_UNITS1=2048
HIDDEN_UNITS2=1024
HIDDEN_UNITS=NUM_FLOWs
LEARNING_RATE=0.00001
EPOCH=1000
VECTOR_SIZE = NUM_FLOWs
print("BATCH_SIZE:",BATCH_SIZE)
print("HIDDEN_UNITS1:",HIDDEN_UNITS1)
print("HIDDEN_UNITS:",HIDDEN_UNITS)
print("LEARNING_RATE:",LEARNING_RATE)


sequence_length = 4*24
from sklearn import preprocessing

def maxminnorm(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        if maxcols[i] == mincols[i]:
            t[:,i] = 0.0
        else:
            t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
            #t[:,i]=(array[:,i])/(maxcols[i])
    return t

def max_norm(array):
    normalised_data = []
    for window in array:
        mx = window.max(axis=0)
        for i in range(mx.shape[0]):
            if mx[i] == 0.0:
                mx[i] = 1.0
        window = window / mx
        normalised_data.append(window)
    return normalised_data

maxcols=TMs.max(axis=0)
mincols=TMs.min(axis=0)
data = maxminnorm(TMs)
#data = TMs
result = []
for index in range(data.shape[0] - sequence_length-1):
    result.append(data[index: index + sequence_length+1])

#result = max_norm(result)

# split into input and outputs
result = np.array(result)

#data = maxminnorm(TMs)





#result = np.array(result)
y = result[:,-1,:]
x = result[:,:-1,:]

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.90, random_state=33)



import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt



TRAIN_EXAMPLES=x_train.shape[0]
TEST_EXAMPLES=x_test.shape[0]

#变厚矩阵
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
#卷积处理 变厚过程
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1] x_movement、y_movement就是步长
    # Must have strides[0] = strides[3] = 1 padding='SAME'表示卷积后长宽不变
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#pool 长宽缩小一倍
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, sequence_length, NUM_FLOWs]) #原始数据的维度：16
ys = tf.placeholder(tf.float32, [None, NUM_FLOWs])#输出数据为维度：1

#------------------------------------construct CNN------------------------------------------#
keep_prob = tf.placeholder(tf.float32)#dropout的比例

x_image = tf.reshape(xs, [-1, NUM_NODE, NUM_NODE, 1])#原始数据144变成二维图片12*12
## conv1 layer ##第一卷积层
W_conv1 = weight_variable([4,4, 1,32]) # patch 2x2, in size 1, out size 32,每个像素变成32个像素，就是变厚的过程
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 2x2x32，长宽不变，高度为32的三维图像
#h_pool1 = max_pool_2x2(h_conv1)     # output size 2x2x32 长宽缩小一倍
print("h_conv1:",h_conv1.shape)
## conv2 layer ##第二卷积层
W_conv2 = weight_variable([4,4, 32, 64]) # patch 2x2, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) #输入第一层的处理结果 输出shape 4*4*64
print("h_conv2:",h_conv2.shape)

## conv1 layer ##第三卷积层
#W_conv3 = weight_variable([4,4, 64,64]) # patch 2x2, in size 1, out size 32,每个像素变成32个像素，就是变厚的过程
#b_conv3 = bias_variable([64])
#h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3) # output size 2x2x32，长宽不变，高度为32的三维图像
#h_pool1 = max_pool_2x2(h_conv1)     # output size 2x2x32 长宽缩小一倍
#print("h_conv3:",h_conv3.shape)
## conv2 layer ##第四卷积层
#W_conv4 = weight_variable([4,4, 64, 64]) # patch 2x2, in size 32, out size 64
#b_conv4 = bias_variable([64])
#h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4) #输入第一层的处理结果 输出shape 4*4*64
#print("h_conv4:",h_conv4.shape)



## fc1 layer ##  full connection 全连接层
W_fc1 = weight_variable([23*23*64, 1024])#4x4 ，高度为64的三维图片，然后把它拉成512长的一维数组
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_conv2, [-1, 23*23*64])#把4*4，高度为64的三维图片拉成一维数组 降维处理
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#把数组中扔掉比例为keep_prob的元素
## fc2 layer ## full connection
W_fc2 = weight_variable([1024, VECTOR_SIZE])#512长的一维数组压缩为长度为1的数组
b_fc2 = bias_variable([VECTOR_SIZE])#偏置
#最后的计算结果
cnn_out =  tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cnn_out = tf.concat([cnn_out,tf.reshape(xs, [-1,NUM_FLOWs])],axis = 1)
print("cnn_out:",cnn_out.shape)
'''
'''
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
#print("cnn_out:",cnn_out.shape)
#dynamic rnn
outputs,states = tf.nn.dynamic_rnn(cell=multi_lstm,inputs=cnn_out,initial_state=init_state,dtype=tf.float32)
W_fc3 = weight_variable([NUM_FLOWs, NUM_FLOWs])#512长的一维数组压缩为长度为1的数组
b_fc3 = bias_variable([NUM_FLOWs])#偏置
outputs = tf.reshape(outputs,[BATCH_SIZE*sequence_length,NUM_FLOWs])
cnn_out =  tf.nn.relu(tf.matmul(outputs, W_fc3) + b_fc3)
cnn_out = tf.reshape(cnn_out,[BATCH_SIZE,sequence_length,NUM_FLOWs])
prediction = cnn_out[:,-1,:]

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
    results = np.zeros(shape=(TEST_EXAMPLES, 1))
    train_losses=[]
    test_losses=[]
    print("epoch:",epoch)
    for j in range(TRAIN_EXAMPLES//BATCH_SIZE):
        _,train_loss=sess.run(
                fetches=(optimizer,mse),
                feed_dict={
                        xs:x_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                        ys:y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE], keep_prob: 0.3
                    }
        )
        train_losses.append(train_loss)
    print("average training loss:", sum(train_losses) / len(train_losses))
    results_train.append(sum(train_losses) / len(train_losses))

    count = 0
    for j in range(TEST_EXAMPLES//BATCH_SIZE):
        result,test_loss=sess.run(
                fetches=(prediction,mse),
                feed_dict={
                        xs:x_test[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                        ys:y_test[j*BATCH_SIZE:(j+1)*BATCH_SIZE], keep_prob: 1.0
                    }
        )
        test_losses.append(test_loss)
        if epoch == EPOCH:
            for tm in result:
                f1 = open('TM_pred_CRNN/'+str(count)+'.txt','w')
                count += 1
                for source_index in range(23):
                    for destination_index in range(23):
                        pos = source_index*23+destination_index
                        f1.write(str(source_index)+' '+str(destination_index)+' '+str(tm[pos]*(maxcols[pos]-mincols[pos])+mincols[pos])+'\n')
                f1.close()
        
    print("average test loss:", sum(test_losses) / len(test_losses))
    results_test.append(sum(test_losses) / len(test_losses))

print("max test mse: ",max(results_test))
print("mean test mse: ",sum(results_test)/len(results_test))
print("min test mse: ",min(results_test))
