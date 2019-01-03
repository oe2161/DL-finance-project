# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 16:54:55 2019

@author: Othman
"""


# Make sure that you have all these libaries available to run the code successfully
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
import random
import time
#%matplotlib inline

tf.reset_default_graph()

## Changing working directory

cwd = os.getcwd()

path='Data for DNN'
os.chdir(path)

cwd = os.getcwd()

## Load csv

file_path = 'ESZ2018.csv'
#filep = 'nanotick.CL/nanotick/2018/11/19/Replay/382.20181118-20181119.1-45919747.log'

#dataframe = pd.read_csv(filepath, sep=",", nrows=2000)
    
start_time = time.time()
# number of previous timestamps used for prediction
n_times = 40
# number of quote levels used for prediction
n_levels = 10
    
split=0.7
test_size=0.2
nrows=1000    
#Labels to extract from data
L_labels = ['Date', 'Time', 'L1-BidPrice', 'L1-BidSize', 'L1-BuyNo', 'L1-AskPrice', 
            'L1-AskSize', 'L1-SellNo', 'L2-BidPrice', 'L2-BidSize', 'L2-BuyNo', 
            'L2-AskPrice', 'L2-AskSize', 'L2-SellNo', 'L3-BidPrice', 'L3-BidSize', 
            'L3-BuyNo', 'L3-AskPrice', 'L3-AskSize', 'L3-SellNo', 'L4-BidPrice', 
            'L4-BidSize', 'L4-BuyNo', 'L4-AskPrice', 'L4-AskSize', 'L4-SellNo', 
            'L5-BidPrice', 'L5-BidSize', 'L5-BuyNo', 'L5-AskPrice', 'L5-AskSize', 
            'L5-SellNo', 'L6-BidPrice', 'L6-BidSize', 'L6-BuyNo', 'L6-AskPrice', 
            'L6-AskSize', 'L6-SellNo', 'L7-BidPrice', 'L7-BidSize', 'L7-BuyNo', 
            'L7-AskPrice', 'L7-AskSize', 'L7-SellNo', 'L8-BidPrice', 'L8-BidSize', 
            'L8-BuyNo', 'L8-AskPrice', 'L8-AskSize', 'L8-SellNo', 'L9-BidPrice', 
            'L9-BidSize', 'L9-BuyNo', 'L9-AskPrice', 'L9-AskSize', 'L9-SellNo', 
            'L10-BidPrice', 'L10-BidSize', 'L10-BuyNo', 'L10-AskPrice', 'L10-AskSize', 
            'L10-SellNo']

# import tick data from the given path
df_import = pd.read_csv(file_path, nrows=nrows)
df = df_import[L_labels]
df2 = pd.DataFrame()
for i in range(n_levels):
    i = i+1
    df2['relative depth'+str(i)]=df['L'+str(i)+'-BidSize']/(df['L'+str(i)+'-BidSize']+df['L'+str(i)+'-AskSize'])
df2['mid price']=(df['L1-AskPrice']+df['L1-BidPrice'])/2
df2['mid_price_change'] = 0
for i in range(1,len(df2)):
    df2.loc[i,'mid_price_change'] = df2.loc[i,'mid price']- df2.loc[i-1,'mid price']
df2['classification'] = df2['mid_price_change'].apply(lambda x: 0 if x == 0 else 1 if x > 0 else -1)
df2 = df2.drop(['mid_price_change'],axis=1)
df2 = df2.drop(['mid price'],axis=1)
# undersampling with temporal bias
# set up a remaining rate
rr = 0.5

# construct bins of the majority class
my_list = []
temp_list = []
remain_index = []
for i in range(len(df2)):
    if df2.loc[i,'classification']==0:
        temp_list.append(i)
    else:
        remain_index.append(i)
        if len(temp_list)>0: 
            my_list.append(temp_list)
            temp_list = []

# in each bin, drop the data with a relative probability according to its position 
for bins in my_list:
    length = len(bins)
    temp_list = np.arange(length)+1
    pp = temp_list/sum(temp_list)
    num = math.ceil(length*rr)
    temp_index = np.random.choice(temp_list,num,p=pp,replace=False)
    for i in temp_index:
        remain_index.append(bins[i-1])
df2 = df2.loc[remain_index]
df2 = df2.sort_index()
df2 = df2.reset_index(drop=True)

    # Oversampling using Smote algorithm
# setup the oversampling/enlarging rate er and the number of nearest neighbors k
er = 0.5
k = 2

# calculate difference between 2 feature vectors
def diff(x,y,df,my_dict):
    x_index = my_dict[x]
    y_index = my_dict[y]
    vec1 = np.array(df2.iloc[x_index,:df.shape[1]-1])
    vec2 = np.array(df2.iloc[y_index,:df.shape[1]-1])
    vec_diff = vec2-vec1
    return vec_diff

# find index of nearest k neighbors from j's row of data
def find_neighbors(j,k,my_dict,my_dict2,df):
    i = my_dict[j]
    global dist_matrix
    for l in range(len(dist_matrix)):
        if dist_matrix[i][l]!=0 or i == l:
            continue
        dist_matrix[i][l]=dist_matrix[l][i]=LA.norm(diff(i,l,df,my_dict2))
    a = dist_matrix[i]
    neighbors = sorted(range(len(a)), key=lambda i: a[i])[:k+1]
    neighbors.remove(i)
    res = []
    for l in neighbors:
        res.append(my_dict2[l])
    return res

# oversample the data with positive mid_price move
temp_list = []
for i in range(len(df2)):
    if df2.loc[i,'classification']==1:
        temp_list.append(i)
temp_list = np.asarray(temp_list)
length = len(temp_list)

# initialize a matrix to save the feature difference between 2 feature vectors
dist_matrix = np.zeros((length,length))
# 2 dictionary to transform between matrix index and dataframe row index
my_dict = {}
my_dict2 = {}
for i in range(length):
    my_dict[temp_list[i]]=i
    my_dict2[i] = temp_list[i]

# main function
num = math.ceil(length*er)
case_index = np.random.choice(temp_list,num,replace=True)
count = 1
for j in case_index:
    top_indexs = find_neighbors(j,k,my_dict,my_dict2,df2)
    top_indexs = np.asarray(top_indexs)
    my_index = np.random.choice(top_indexs)
    temp_array = np.array(df2.iloc[j,:df2.shape[1]-1])
    difference = diff(my_dict[j],my_dict[my_index],df2,my_dict2)
    ran_num = random.uniform(0, 1)
    temp_array = temp_array+difference*ran_num
    temp_array = temp_array.tolist()
    temp_array.append(1)
    df2.loc[j+count/(num+1)] = temp_array
    count += 1
df2 = df2.sort_index()
df2 = df2.reset_index(drop=True)
# oversample the data with negative mid_price move
temp_list = []
for i in range(len(df2)):
    if df2.loc[i,'classification']==-1:
        temp_list.append(i)
temp_list = np.asarray(temp_list)
length = len(temp_list)

# initialize a matrix to save the feature difference between 2 feature vectors
dist_matrix = np.zeros((length,length))
# 2 dictionary to transform between matrix index and dataframe row index
my_dict = {}
my_dict2 = {}
for i in range(length):
    my_dict[temp_list[i]]=i
    my_dict2[i] = temp_list[i]

# main function
num = math.ceil(length*er)
case_index = np.random.choice(temp_list,num,replace=True)
count = 1
for j in case_index:
    top_indexs = find_neighbors(j,k,my_dict,my_dict2,df2)
    top_indexs = np.asarray(top_indexs)
    my_index = np.random.choice(top_indexs)
    temp_array = np.array(df2.iloc[j,:df2.shape[1]-1])
    difference = diff(my_dict[j],my_dict[my_index],df2,my_dict2)
    ran_num = random.uniform(0, 1)
    temp_array = temp_array+difference*ran_num
    temp_array = temp_array.tolist()
    temp_array.append(-1)
    df2.loc[j+count/(num+1)] = temp_array
    count += 1
df2 = df2.sort_index()
df2 = df2.reset_index(drop=True)

df2['classification'] = df2['classification'].apply(lambda x: [0,1,0] if x == 0 else [1,0,0] if x > 0 else [0,0,1])
#df3['classification']=df3['classification'].reshape
#Splitting the data into a training and test data:
y = df2['classification'].values
y=np.stack(y)
X = df2.drop('classification', axis=1).values
train_data, test_data, train_label, test_label = train_test_split(X, y, test_size = test_size, random_state=42, stratify=y)
#y_test=y_test.reshape(X_test.shape[0],3)
num_examples = train_data.shape[0] 




EMA = 0.0
gamma = 0.1
for ti in range(num_examples):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA

# Used for visualization and test purposes
#all_mid_data = np.concatenate([train_data,test_data],axis=0)


epochs_completed = 0
index_in_epoch = 0

    # for splitting out batches of data
def next_batch(batch_size):

    global train_data
    global train_label
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_data = train_data[perm]
        train_label = train_label[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_data[start:end], train_label[start:end]

def unroll_batch(batch_size):
    global num_unrollings
    train,label=[],[]
    for i in range(num_unrollings):
        train_temp_data,train_temp_label=next_batch(batch_size)
        train.append(train_temp_data)
        label.append(train_temp_label)
    return train, label
    
D = 10 # Dimensionality of the data.
num_unrollings = 40 # Number of time steps you look into the future.
batch_size = 200 # Number of samples in a batch
num_nodes = [400,300,200,200,150] # Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = 5 # number of layers
dropout = 0.2 # dropout amount

tf.reset_default_graph() # This is important in case you run this multiple times

# Input data.
train_inputs, train_outputs = [],[]

# You unroll the input over time defining placeholders for each time step
for ui in range(num_unrollings):
    train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size,D],name='train_inputs_%d'%ui))
    train_outputs.append(tf.placeholder(tf.float32, shape=[batch_size,3], name = 'train_outputs_%d'%ui))


lstm_cells = [
    tf.contrib.rnn.LSTMCell(num_units=num_nodes[li],
                            state_is_tuple=True,
                            initializer= tf.contrib.layers.xavier_initializer()
                           )
 for li in range(n_layers)]

drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
    lstm, input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout
) for lstm in lstm_cells]
drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

w = tf.get_variable('w',shape=[num_nodes[-1], 3], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable('b',initializer=tf.random_uniform([3],-0.1,0.1))



c, h = [],[]
initial_state = []
for li in range(n_layers):
  c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
  h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
  initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))

# Do several tensor transofmations, because the function dynamic_rnn requires the output to be of
# a specific format. Read more at: https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs],axis=0)

# all_outputs is [seq_length, batch_size, num_nodes]
all_lstm_outputs, state = tf.nn.dynamic_rnn(
    drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
    time_major = True, dtype=tf.float32)

all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size*num_unrollings,num_nodes[-1]])

all_outputs = tf.nn.xw_plus_b(all_lstm_outputs,w,b)


split_outputs = tf.split(all_outputs,num_unrollings,axis=0)

#prediction = tf.nn.softmax(split_outputs)



# When calculating the loss you need to be careful about the exact form, because you calculate
# loss of all the unrolled steps at the same time
# Therefore, take the mean error or each batch and get the sum of that over all the unrolled steps

print('Defining training Loss')
loss = 0.0
with tf.control_dependencies([tf.assign(c[li], state[li][0]) for li in range(n_layers)]+
                             [tf.assign(h[li], state[li][1]) for li in range(n_layers)]):
  for ui in range(num_unrollings):
    loss += tf.reduce_mean(0.5*(split_outputs[ui]-train_outputs[ui])**2)

print('Learning rate decay operations')
global_step = tf.Variable(0, trainable=False)
inc_gstep = tf.assign(global_step,global_step + 1)
tf_learning_rate = tf.placeholder(shape=None,dtype=tf.float32)
tf_min_learning_rate = tf.placeholder(shape=None,dtype=tf.float32)

learning_rate = tf.maximum(
    tf.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5, staircase=True),
    tf_min_learning_rate)

# Optimizer.
print('TF Optimization operations')
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, v = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimizer = optimizer.apply_gradients(
    zip(gradients, v))

print('\tAll done')




print('Defining prediction related TF functions')

sample_inputs = tf.placeholder(tf.float32, shape=[1,D])
y = tf.placeholder(tf.float32, shape=[1,3])
# Maintaining LSTM state for prediction stage
sample_c, sample_h, initial_sample_state = [],[],[]
for li in range(n_layers):
  sample_c.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
  sample_h.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
  initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(sample_c[li],sample_h[li]))

reset_sample_states = tf.group(*[tf.assign(sample_c[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)],
                               *[tf.assign(sample_h[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])

sample_outputs, sample_state = tf.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs,0) ,
                                   initial_state=tuple(initial_sample_state),
                                   time_major = True,
                                   dtype=tf.float32)

with tf.control_dependencies([tf.assign(sample_c[li],sample_state[li][0]) for li in range(n_layers)]+
                              [tf.assign(sample_h[li],sample_state[li][1]) for li in range(n_layers)]):  
  sample_prediction = tf.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]), w, b)
  sample_prediction = tf.nn.softmax(sample_prediction)

print('\tAll done')




epochs = 20
valid_summary = 1 # Interval you make test predictions

n_predict_once = 10 # Number of steps you continously predict for

train_seq_length = train_data.size # Full length of the training data

train_mse_ot = [] # Accumulate Train losses
test_mse_ot = [] # Accumulate Test loss
predictions_over_time = [] # Accumulate predictions

session = tf.InteractiveSession()

tf.global_variables_initializer().run()

# Used for decaying learning rate
loss_nondecrease_count = 0
loss_nondecrease_threshold = 2 # If the test error hasn't increased in this many steps, decrease learning rate

print('Initialized')
average_loss = 0


x_axis_seq = []

# Points you start your test predictions from
test_points_seq = np.arange(1,num_examples,50).tolist()

for ep in range(epochs):       

    # ========================= Training =====================================
    for step in range(train_seq_length//batch_size):

        u_data, u_labels = unroll_batch(batch_size)
        feed_dict = {}
        for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):            
            feed_dict[train_inputs[ui]] = dat
            feed_dict[train_outputs[ui]] = lbl

        feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate:0.000001})

        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += l

    # ============================ Validation ==============================
    if (ep+1) % valid_summary == 0:

      average_loss = average_loss/(valid_summary*(train_seq_length//batch_size))

      # The average loss
      if (ep+1)%valid_summary==0:
        print('Average loss at step %d: %f' % (ep+1, average_loss))

      train_mse_ot.append(average_loss)

      average_loss = 0 # reset loss


print('\tFinished Predictions')
      
      
      
      
      
