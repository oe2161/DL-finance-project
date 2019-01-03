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
import os
import numpy as np
import pandas as pd
import numpy as np
import math
import random
from numpy import linalg as LA
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split




tf.reset_default_graph()
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


## Changing working directory

cwd = os.getcwd()

Path = input("What path are we going to use : ")
if (Path == "VM") :
    path = '/home/ja3291/'
    file_path = 'ESZ2018.csv'
elif (Path == ""):
    path = '/Users/ayoujiljad/Documents/Python/2018212-12-06-2018_3_12_22/'
    file_path = 'ESZ2018.csv'
else:
    file_path = Path 

os.chdir(path)

cwd = os.getcwd()

# 256 neurons in each hidden layers
n_hidden_1 = 800
n_hidden_2 = 700
n_hidden_3 = 600
n_hidden_4 = 500
n_hidden_5 = 400
n_hidden_6 = 300
n_hidden_7 = 150
n_hidden_8 = 50

# There are 10 levels, and we consider 40 timestamps and the mid_price
# output size
input_size = 800
output_size = 3


# Parameters
learning_rate = 0.001
training_epochs = 2500
batch_size = 100
display_step = 1
test_size=0.2

file_path = 'ESZ2018.csv'
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
Nrow = input("How many rows in the dataset do you want to consider ?")
if (Nrow == ''):
    df_import = pd.read_csv(file_path)
else : 
    df_import = pd.read_csv(file_path, nrows=int(Nrow))    

df = df_import[L_labels]
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
    df2['relative_price'+str(i)]=(df['L'+str(i)+'-BidPrice']-df['L'+str(i)+'-AskPrice'])/df['L'+str(i)+'-AskPrice']
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
    dataf = df2.iloc[i:(i+n_times),0:20]
    datf = dataf.values.reshape(dataf.shape[0]*dataf.shape[1])
    df_temp = pd.DataFrame([list(datf)])
    df_temp['mid_price_change']=df2.loc[i+n_times,'mid_price_change']
    df3 = pd.concat([df3,df_temp],ignore_index=True)
print("   Done..       ")
df3['classification'] = 0
df3['classification'] = df3['mid_price_change'].apply(lambda x: 0 if x == 0 else 1 if x > 0 else -1)
df3 = df3.drop(['mid_price_change'],axis=1)
print("Done.")
print()


# undersampling with temporal bias
print("Random undersampling...")
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
print("Done.")
print()

    # Oversampling using Smote algorithm
# setup the oversampling/enlarging rate er and the number of nearest neighbors k
k = 2
print("Smote Algorithm Oversampling...")

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

temp_list = []
temp_list_neg = []
for i in range(len(df3)):
    if df3.loc[i,'classification']==1:
        temp_list.append(i)
    if df3.loc[i,'classification']==-1:
        temp_list_neg.append(i)
temp_list = np.asarray(temp_list)
length = len(temp_list)
temp_list_neg = np.asarray(temp_list_neg)
length_neg = len(temp_list_neg)

num_pos = max(1,len(df3)-length)
num_neg = max(1,len(df3)-length_neg)

# oversample the data with positive mid_price move

# initialize a matrix to save the feature difference between 2 feature vectors
dist_matrix = np.zeros((length,length))
# 2 dictionary to transform between matrix index and dataframe row index
my_dict = {}
my_dict2 = {}
for i in range(length):
    my_dict[temp_list[i]]=i
    my_dict2[i] = temp_list[i]

# main function
case_index = np.random.choice(temp_list,num_pos,replace=True)
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

temp_list = temp_list_neg
length = length_neg

# initialize a matrix to save the feature difference between 2 feature vectors
dist_matrix = np.zeros((length,length))
# 2 dictionary to transform between matrix index and dataframe row index
my_dict = {}
my_dict2 = {}
for i in range(length):
    my_dict[temp_list[i]]=i
    my_dict2[i] = temp_list[i]

# main function
case_index = np.random.choice(temp_list,num_neg,replace=True)
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
print("Done.")
print()

df3['classification'] = df3['classification'].apply(lambda x: [0,1,0] if x == 0 else [1,0,0] if x > 0 else [0,0,1])
#df3['classification']=df3['classification'].reshape
#Splitting the data into a training and test data:
y = df3['classification'].values
y=np.stack(y)
X = df3.drop('classification', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=42, stratify=y)
num_examples = X_train.shape[0]


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
    with tf.variable_scope("hidden_layer_6"):
        hidden_6 = layer2(hidden_5, [n_hidden_5, n_hidden_6], [n_hidden_6])
        
    with tf.variable_scope("hidden_layer_7"):
        hidden_7 = layer2(hidden_6, [n_hidden_6, n_hidden_7], [n_hidden_7])
        
    with tf.variable_scope("hidden_layer_8"):
        hidden_8 = layer2(hidden_7, [n_hidden_7, n_hidden_8], [n_hidden_8])    
    
    with tf.variable_scope("output"):
        output = layer1(hidden_8, [n_hidden_8, output_size], [output_size])
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
    return X_train[start:end], y_train[start:end]





if __name__ == '__main__':
    
    with tf.Graph().as_default():
        
        with tf.variable_scope("MNIST_convoultional_model"):
            #neural network definition
            
            #the input variables are first define as placeholder 
            # a placeholder is a variable/data which will be assigned later 
            # MNIST data image of shape 28*28=784
            x = tf.placeholder("float", [None, 800]) 
            # 0-9 digits recognition
            y = tf.placeholder("float", [None,3])  
            
            # dropout probability
    #            keep_prob = tf.placeholder(tf.float32) 
            #the network is defined using the inference function defined above in the code
            output = inference(x)#, keep_prob)
            cost = loss(output, y)
            #initialize the value of the global_step variable 
            # recall: it is incremented by one each time the .minimise() is called
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = training(cost, global_step)
            #evaluate the accuracy of the network (done on a validation set)
            eval_op = evaluate(output, y)
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()
            sess = tf.Session()
            
            summary_writer = tf.summary.FileWriter(file_path, sess.graph)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            
            # Training cycle
            for epoch in range(training_epochs):
    
                avg_cost = 0.0
                total_batch = int(len(df3)/batch_size)
                
                # Loop over all batches
                for i in range(total_batch):
                    
                    
                    
                    minibatch_x, minibatch_y = next_batch(batch_size)
                    #minibatch_y=minibatch_y.reshape(batch_size ,1)
                    # Fit training using batch data
                    sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
                    
                    # Compute average loss
                    avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y})/total_batch
                
                
                # Display logs per epoch step
                if epoch % display_step == 0:
                    
                    print("Epoch:", '%04d' % (epoch+1), "cost =", "{:0.9f}".format(avg_cost))
                    
                    #probability dropout of 1 during validation
                    accuracy = sess.run(eval_op, feed_dict={x: minibatch_x, y: minibatch_y})
                    print("Validation Error:", (1 - accuracy))
                    
                    # probability dropout of 0.25 during training
                    summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
                    summary_writer.add_summary(summary_str, sess.run(global_step))
                    
                    #saver.save(sess, file_path+'model-checkpoint', global_step=global_step)
                    saver.save(sess, '/Users/ayoujiljad/Documents/Python/2018212-12-06-2018_3_12_22/test_model')
                    
            print("Optimization Done")
                    
            accuracy = sess.run(eval_op, feed_dict={x: X_test, y: y_test})
            print("Test Accuracy:", accuracy)
                    
        elapsed_time = time.time() - start_time
        print('Execution time was %0.3f' % elapsed_time)