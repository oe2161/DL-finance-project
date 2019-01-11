#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 21:49:43 2019

@author: ayoujiljad
"""

import numpy as np

from keras.layers.core import Activation
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
import keras
from ntm import NeuralTuringMachine as NTM
import tensorflow as tf
import time
import os
import numpy as np
import pandas as pd
import numpy as np
import math
import random
from numpy import linalg as LA
from sklearn.model_selection import train_test_split

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
df2['mid price']=(df['L1-AskPrice']+df['L1-BidPrice'])/2
df2['mid_price_change'] = 0
print("   Done..       ")
print("   Second step..")
count = 0
total = len(df2)
for i in range(1,total):
    if (count/10 < np.floor(10*i/total)):
        count = round(100*i/total)
        print("    "+str(count)+" %")
    df2.loc[i,'mid_price_change'] = df2.loc[i,'mid price']- df2.loc[i-1,'mid price']
print("   Done..       ")
print("   Third step..")
df3 = pd.DataFrame()
count = 0
for i in range(total-n_times):
    if (count/10 < np.floor(10*i/(total-n_times))):
        count = round(100*i/(total-n_times))
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

num_pos = int(max(1,0.125*len(df3)-length))
num_neg = int(max(1,0.125*len(df3)-length_neg))

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
df3.to_csv(path_or_buf=path+'dataframe_save')
#df3['classification']=df3['classification'].reshape
#Splitting the data into a training and test data:
y = df3['classification'].values
y=np.stack(y)
X = df3.drop('classification', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=42, stratify=y)
#y_test=y_test.reshape(X_test.shape[0],3)

####
#### Running the training of the NTM and test (accuracy)
####

input_dim = np.shape()[0]
n_slots = 128
m_depth = 20
learning_rate = 5e-4
clipnorm = 10

model = Sequential()
model.name = "NTM_-_" + None.name
model.batch_size = batch_size
model.input_dim = input_dim
model.output_dim = 1

ntm = NTM(1, n_slots=n_slots, #n_slots: Memory width
          m_depth=m_depth, #m_depth: Memory depth at each location
          shift_range=3,   
          #shift_range: int, number of available shifts, ex. if 3, avilable shifts are
          #                 (-1, 0, 1)
          controller_model=None,
          #controller_model: A keras model with required restrictions to be used as a controller.
          #                  The requirements are appropriate shape, linear activation and stateful=True if recurrent.
          #                  Default: One dense layer.
          activation="sigmoid",
          #activation: This is the activation applied to the layer output.
          #            It can be either a Keras activation or a string like "tanh", "sigmoid", "linear" etc.
          #            Default is linear.
          read_heads = 1,write_heads = 1,return_sequences=True,
          input_shape=(None, input_dim),batch_size = batch_size)

model.add(ntm)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate, clipnorm=clipnorm), metrics = ['binary_accuracy'], sample_weight_mode="temporal")
model.fit(X_train, y_train, epochs=500, verbose=0)
y_pred = model.predict_classes(X_test)
accuracy = keras.metrics.categorical_accuracy(y_test, y_pred)
print("Accuracy : "+accuracy)
