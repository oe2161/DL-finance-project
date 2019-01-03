#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:26:13 2019

@author: ayoujiljad
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 19:12:51 2018

@author: ayoujiljad
"""

# -*- coding: utf-8 -*-
"""
Éditeur de Spyder
Ceci est un script temporaire.
"""

import tensorflow as tf
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import numpy as np
import math
import random
import seaborn as sns
from numpy import linalg as LA
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split


## Changing working directory

cwd = os.getcwd()

start = 20000

Path = input("What path are we going to use : ")
if (Path == "VM") :
    path = '/home/ja3291/'
    file_path = 'ESZ2018.csv'
elif (Path == ""):
    path = '/Users/ayoujiljad/Documents/Python/2018212-12-06-2018_3_12_22/'
    file_path = 'ESZ2018.csv'
else:
    file_path = Path 
    
Path_r = input("What path are we going to restore the model from : ")
if (Path_r == "VM") :
    rest = '/home/ja3291/'
elif (Path_r == ""):
    rest = '/Users/ayoujiljad/Documents/Python/2018212-12-06-2018_3_12_22/test_model'
else:
    rest = Path_r 

os.chdir(path)

cwd = os.getcwd()

# 256 neurons in each hidden layers
n_hidden_1 = 400
n_hidden_2 = 300
n_hidden_3 = 200
n_hidden_4 = 100
n_hidden_5 = 50

# There are 10 levels, and we consider 40 timestamps and the mid_price
# output size
input_size = 400
output_size = 3

# Parameters
learning_rate = 0.001
training_epochs = 2500
batch_size = 32
display_step = 1
test_size=0.2

start_time = time.time()
# number of previous timestamps used for prediction
n_times = 40
# number of quote levels used for prediction
n_levels = 10

split=0.7

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
print("Importing the data...")
Nrow = int(input("How many rows in the dataset do you want to consider ?"))
if (Nrow == ''):
    df_import = pd.read_csv(file_path)
else : 
    df_import_1 = pd.read_csv(file_path, nrows=int(Nrow+start))  
    df_import = df_import_1.tail(Nrow)
    df_import = df_import.reset_index(drop =True)

df = df_import[L_labels]

L_prices = ['L1-BidPrice', 'L1-AskPrice', 'L2-BidPrice','L2-AskPrice','L3-BidPrice',
            'L3-AskPrice', 'L4-BidPrice', 'L4-AskPrice', 'L5-BidPrice', 'L5-AskPrice',
            'L6-BidPrice', 'L6-AskPrice', 'L7-BidPrice', 'L7-AskPrice','L8-BidPrice',
            'L8-AskPrice','L9-BidPrice','L9-AskPrice','L10-BidPrice','L10-AskPrice']

L_p = np.unique(df.loc[:, L_prices].values)
min_p = min(L_p)
max_p = max(L_p)

Prices = list(np.arange(min(L_p),max(L_p)+0.25,0.25))[::-1]
print(Prices)
heat_map = np.zeros((Nrow,len(Prices)))
for i in range(Nrow):
    for j in range(1,n_levels+1):
        i_p = Prices.index(df.loc[i,'L'+str(j)+'-AskPrice'])
        j_p = Prices.index(df.loc[i,'L'+str(j)+'-BidPrice'])
        heat_map[i,i_p] = df.loc[i,'L'+str(j)+'-AskSize']
        heat_map[i,j_p] = df.loc[i,'L'+str(j)+'-BidSize']
        
ax = sns.heatmap(np.transpose(heat_map))
plt.xticks(np.arange(len(Prices)), Prices)
#plt.imshow(np.transpose(heat_map), cmap='hot',interpolation='nearest')
plt.show()
    
print("Done.")
df2 = pd.DataFrame()
print()
print("Rearraging the data to compute the mid prices..")
print("   First step..")
count = 0
for i in range(n_levels):
    if (count/10 < np.floor(10*i/n_levels)):
        count = round(100*i/n_levels)
        print ("    "+str(count)+" %"),
    i = i+1
    df2['relative depth'+str(i)]=df['L'+str(i)+'-BidSize']/(df['L'+str(i)+'-BidSize']+df['L'+str(i)+'-AskSize'])
df2['mid price']=(df['L1-AskPrice']+df['L1-BidPrice'])/2
df2['mid_price_change'] = 0
print("   Done..       ")
print("   Second step..")
count = 0
for i in range(1,len(df2)):
    if (count/10 < np.floor(10*i/len(df2))):
        count = round(100*i/len(df2))
        print("    "+str(count)+" %")
    df2.loc[i,'mid_price_change'] = df2.loc[i,'mid price']- df2.loc[i-1,'mid price']
print("   Done..       ")
print("   Third step..")
df3 = pd.DataFrame()
count = 0
for i in range(len(df2)-n_times):
    if (count/10 < np.floor(10*i/(len(df2)-n_times))):
        count = round(100*i/(len(df2)-n_times))
        print("    "+str(count)+" %")
    my_list = []
    dataf = df2.iloc[i:(i+n_times),0:10]
    datf = dataf.values.reshape(dataf.shape[0]*dataf.shape[1])
    df_temp = pd.DataFrame([list(datf)])
    df_temp['mid_price_change']=df2.loc[i+n_times,'mid_price_change']
    df3 = pd.concat([df3,df_temp],ignore_index=True)
print("   Done..       ")
df3['classification'] = 0
df3['classification'] = df3['mid_price_change'].apply(lambda x: 0 if x == 0 else 1 if x > 0 else -1)
df3 = df3.drop(['mid_price_change'],axis=1)
y_test_plot = df3['classification'].values
print("Done.")
print()
print(y_test_plot)

df3['classification'] = df3['classification'].apply(lambda x: [0,1,0] if x == 0 else [1,0,0] if x > 0 else [0,0,1])
#df3['classification']=df3['classification'].reshape
#Splitting the data into a training and test data:
y = df3['classification'].values
y_train=np.stack(y)
X = df3.drop('classification', axis=1).values
X_test_plot = X

l = len(y_test_plot)

X = np.arange(0,l)
Y = 0*X
V = 0*y_test_plot
ax = plt.axes()
ax.set_xlim(left=0, right=len(y_test_plot), xmin=0, xmax=len(y_test_plot))
ax.set_ylim(bottom=-2, top=2, ymin=-2, ymax=2)
for i in range(len(y_test_plot)):
    if (y_test_plot[i] == 1):
        ax.arrow(X[i], 0, 0, y_test_plot[i], head_width=50, head_length=0.1, color = 'g', width = 5)
    elif (y_test_plot[i] == -1):
        ax.arrow(X[i], 0, 0, y_test_plot[i], head_width=50, head_length=0.1, color = 'r', width = 5)
plt.show()





def layer1(x, weight_shape, bias_shape):
    """
    Defines the network layers
    input:
        - x: input vector of the layer
        - weight_shape: shape the the weight maxtrix
        - bias_shape: shape of the bias vector
    output:
        - output vector of the layer after the matrix multiplication and transformation
    """
    
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    
    return tf.nn.softmax(tf.matmul(x, W) + b)


def layer2(x, weight_shape, bias_shape):
    """
    Defines the network layers
    input:
        - x: input vector of the layer
        - weight_shape: shape the the weight maxtrix
        - bias_shape: shape of the bias vector
    output:
        - output vector of the layer after the matrix multiplication and transformation
    """
    
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    
    return tf.nn.relu(tf.matmul(x, W) + b)


def inference(x):
    """
    define the whole network (5 hidden layers + output layers)
    input:
        - a batch of pictures 
        (input shape = (batch_size*image_size))
    output:
        - a batch vector corresponding to the logits predicted by the network
        (output shape = (batch_size*output_size)) 
    """
    print(type(x))
    print(np.shape(x))
    print(x)
    
    with tf.variable_scope("hidden_layer_1"):
        hidden_1 = layer2(x, [input_size, n_hidden_1], [n_hidden_1])
        #print([input_size, n_hidden_1])
     
    with tf.variable_scope("hidden_layer_2"):
        hidden_2 = layer2(hidden_1, [n_hidden_1, n_hidden_2], [n_hidden_2])
        #print([n_hidden_1, n_hidden_2])
        
    with tf.variable_scope("hidden_layer_3"):
        hidden_3 = layer2(hidden_2, [n_hidden_2, n_hidden_3], [n_hidden_3])
        #print([n_hidden_2, n_hidden_3])
        
    with tf.variable_scope("hidden_layer_4"):
        hidden_4 = layer2(hidden_3, [n_hidden_3, n_hidden_4], [n_hidden_4])
        #print([n_hidden_3, n_hidden_4])
        
    with tf.variable_scope("hidden_layer_5"):
        hidden_5 = layer2(hidden_4, [n_hidden_4, n_hidden_5], [n_hidden_5])
        #print([n_hidden_4, n_hidden_5])
     
    with tf.variable_scope("output"):
        output = layer1(hidden_5, [n_hidden_5, output_size], [output_size])
        #print([n_hidden_5, output_size])

    return output


def loss(output, y):
    """
    Computes softmax cross entropy between logits and labels and then the loss 
    
    intput:
        - output: the output of the inference function 
        - y: true value of the sample batch
        
        the two have the same shape (batch_size * num_of_classes)
    output:
        - loss: loss of the corresponding batch (scalar tensor)
    
    """
    #Computes softmax cross entropy between logits and labels.
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)

    return loss

def training(cost, global_step):
    """
    defines the necessary elements to train the network
    
    intput:
        - cost: the cost is the loss of the corresponding batch
        - global_step: number of batch seen so far, it is incremented by one each time the .minimize() function is called
    """
    tf.summary.scalar("cost", cost)
    # using Adam Optimizer 
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output, y):
    """
    evaluates the accuracy on the validation set 
    input:
        -output: prediction vector of the network for the validation set
        -y: true value for the validation set
    output:
        - accuracy: accuracy on the validation set (scalar between 0 and 1)
    """
    #correct prediction is a binary vector which equals one when the output and y match
    #otherwise the vector equals 0
    #tf.cast: change the type of a tensor into another one
    #then, by taking the mean of the tensor, we directly have the average score, so the accuracy
    
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("validation_error", (1.0 - accuracy))
    return accuracy


epochs_completed = 0
index_in_epoch = 0
num_examples = X_test_plot.shape[0]

if __name__ == '__main__':
    
        
    with tf.Graph().as_default():

        with tf.variable_scope("MNIST_convoultional_model"):
            x = tf.placeholder("float", [None, 400]) 
            y = tf.placeholder("float", [None,3])  
            output = inference(x)
            cost = loss(output, y)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = training(cost, global_step)
            eval_op = evaluate(output, y)
            summary_op = tf.summary.merge_all()
            sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess, rest)
            
            summary_writer = tf.summary.FileWriter(file_path, sess.graph)
            init_op = tf.global_variables_initializer()
            
            test_y = sess.run(output, feed_dict={x:X_test_plot, y:y_train})
            
            accuracy = sess.run(eval_op, feed_dict={x: X_test_plot, y: y_train})
            print("Validation Error:", (1 - accuracy))
        elapsed_time = time.time() - start_time
        print('Execution time was %0.3f' % elapsed_time)
        
l = test_y.shape[0]
U = np.zeros(l)
for i in range(l):
    if (test_y[i][0]>test_y[i][1] and test_y[i][0]>test_y[i][2]):
        U[i] == 1
    if (test_y[i][2]>test_y[i][1] and test_y[i][2]>test_y[i][0]):
        U[i] == -1

X = np.arange(0,len(U))
Y = 0*X
V = 0*U
ax = plt.axes()
ax.set_xlim(left=0, right=len(U), xmin=0, xmax=len(U))
ax.set_ylim(bottom=-2, top=2, ymin=-2, ymax=2)
#ax.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
for i in range(len(U)):
    if (U[i] == 1):
        ax.arrow(X[i], 0, 0, U[i], head_width=1, head_length=0.1, color = 'g')
    elif (U[i] == -1):
        ax.arrow(X[i], 0, 0, U[i], head_width=1, head_length=0.1, color = 'r')
plt.show()


