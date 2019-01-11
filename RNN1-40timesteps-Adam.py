""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import os
import numpy as np
import math
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

'''
A one layer RNN with LSTM cell top predict mid-price movement of the limit order book using different time steps
'''

tf.reset_default_graph()


# Training Parameters
learning_rate = 0.001
training_steps = 400
batch_size = 400
display_step = 200

# Network Parameters
num_input = 10 # There are 10 level in the order book
timesteps = 40 # timesteps
num_hidden = 400 # hidden layer num of features
num_classes = 3
dropout = 0.1
nrows=5000


cwd = os.getcwd()

path='Data for DNN'
os.chdir(path)

cwd = os.getcwd()

file_path = 'ESZ2018.csv'
start_time = time.time()
# number of previous timestamps used for prediction
n_times = 40
# number of quote levels used for prediction
n_levels = 10

split=0.7
test_size=0.2

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
df3 = pd.DataFrame()
for i in range(len(df2)-n_times):
    my_list = []
    for j in range(n_times):
        for k in range(n_levels):
            k = k+1
            my_list.append(df2.loc[i+j,'relative depth'+str(k)])
    df_temp = pd.DataFrame([my_list])
    df_temp['mid_price_change']=df2.loc[i+n_times,'mid_price_change']
    df3 = pd.concat([df3,df_temp],ignore_index=True)
df3['classification'] = 0
df3['classification'] = df3['mid_price_change'].apply(lambda x: 0 if x == 0 else 1 if x > 0 else -1)
df3 = df3.drop(['mid_price_change'],axis=1)

# undersampling with temporal bias
# set up a remaining rate
rr = 0.5

# construct bins of the majority class
my_list = []
temp_list = []
remain_index = []
for i in range(len(df3)):
    if df3.loc[i,'classification']==0:
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
df3 = df3.loc[remain_index]
df3 = df3.sort_index()
df3 = df3.reset_index(drop=True)

    # Oversampling using Smote algorithm
# setup the oversampling/enlarging rate er and the number of nearest neighbors k
er = 0.5
k = 2

# calculate difference between 2 feature vectors
def diff(x,y,df,my_dict):
    x_index = my_dict[x]
    y_index = my_dict[y]
    vec1 = np.array(df3.iloc[x_index,:df.shape[1]-1])
    vec2 = np.array(df3.iloc[y_index,:df.shape[1]-1])
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
for i in range(len(df3)):
    if df3.loc[i,'classification']==1:
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
    top_indexs = find_neighbors(j,k,my_dict,my_dict2,df3)
    top_indexs = np.asarray(top_indexs)
    my_index = np.random.choice(top_indexs)
    temp_array = np.array(df3.iloc[j,:df3.shape[1]-1])
    difference = diff(my_dict[j],my_dict[my_index],df3,my_dict2)
    ran_num = random.uniform(0, 1)
    temp_array = temp_array+difference*ran_num
    temp_array = temp_array.tolist()
    temp_array.append(1)
    df3.loc[j+count/(num+1)] = temp_array
    count += 1
df3 = df3.sort_index()
df3 = df3.reset_index(drop=True)
# oversample the data with negative mid_price move
temp_list = []
for i in range(len(df3)):
    if df3.loc[i,'classification']==-1:
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
    top_indexs = find_neighbors(j,k,my_dict,my_dict2,df3)
    top_indexs = np.asarray(top_indexs)
    my_index = np.random.choice(top_indexs)
    temp_array = np.array(df3.iloc[j,:df3.shape[1]-1])
    difference = diff(my_dict[j],my_dict[my_index],df3,my_dict2)
    ran_num = random.uniform(0, 1)
    temp_array = temp_array+difference*ran_num
    temp_array = temp_array.tolist()
    temp_array.append(-1)
    df3.loc[j+count/(num+1)] = temp_array
    count += 1
df3 = df3.sort_index()
df3 = df3.reset_index(drop=True)

df3['classification'] = df3['classification'].apply(lambda x: [0,1,0] if x == 0 else [1,0,0] if x > 0 else [0,0,1])
#df3['classification']=df3['classification'].reshape
#Splitting the data into a training and test data:
y = df3['classification'].values
y=np.stack(y)
X = df3.drop('classification', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=42, stratify=y)
#y_test=y_test.reshape(X_test.shape[0],3)
num_examples = X_train.shape[0]   



# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    drop_lstm_cell=tf.contrib.rnn.DropoutWrapper(
    lstm_cell, input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(drop_lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer you can try either Adam or SGD with clipping
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#gradients, v = zip(*optimizer.compute_gradients(loss_op))
#gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
#optimizer = optimizer.apply_gradients(
#    zip(gradients, v))


# Evaluate model 
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


epochs_completed = 0
index_in_epoch = 0

# for splitting out batches of data
def next_batch(batch_size):

    global X_train
    global y_train
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
        X_train = X_train[perm]
        y_train = y_train[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return X_train[start:end], y_train[start+n_times:end+n_times]





# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = next_batch(batch_size)
        # Reshape data to get 40 time step to predict the future.
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    #Testing the predictive power of the model
    test_data = X_test.reshape((-1, timesteps, num_input))
    test_data=test_data[:-40]
    test_label = y_test[40:len(y_test)]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
    pred=sess.run(prediction, feed_dict={X:test_data,Y:test_label})

#Plotting the probability values returned by the trained model on the test set
ups_pred=[]
down_pred=[]
no_mv_pred=[]    
for i in range(len(test_data)):
    if np.argmax(test_label[i])==0:
        ups_pred.append(pred[i][0])
    elif np.argmax(test_label[i])==2:
        down_pred.append(pred[i][2])
    else:
        no_mv_pred.append(pred[i][1])
        
        
plt.plot(ups_pred)
plt.title('Probability of Mid_price=1')
plt.show()

plt.plot(down_pred)
plt.title('Probability of Mid_price=-1')
plt.show()


plt.plot(down_pred)
plt.title('Probability of Mid_price=0')
plt.show()
