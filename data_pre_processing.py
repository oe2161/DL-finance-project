
# coding: utf-8

# In[258]:


import pandas as pd
import numpy as np
import math
import random
from numpy import linalg as LA

# number of previous days used for prediction
n_days = 10
# number of quote levels used for prediction
n_levels = 5

# import tick data from excel
df = pd.read_excel('FUTURES ORDER BOOK.xlsx')
df2 = pd.DataFrame()
for i in range(n_levels):
    i = i+1
    df2['relative depth'+str(i)]=df['BidSize'+str(i)]/(df['BidSize'+str(i)]+df['AskSize'+str(i)])
df2['mid price']=(df['Ask1']+df['Bid1'])/2
df2['mid_price_change'] = 0
for i in range(1,len(df_new)):
    df2.loc[i,'mid_price_change'] = df2.loc[i,'mid price']- df2.loc[i-1,'mid price']
df3 = pd.DataFrame()
for i in range(len(df_new)-n_days):
    my_list = []
    for j in range(n_days):
        for k in range(n_levels):
            k = k+1
            my_list.append(df2.loc[i+j,'relative depth'+str(k)])
    df_temp = pd.DataFrame([my_list])
    df_temp['mid_price_change']=df2.loc[i+n_days,'mid_price_change']
    df3 = pd.concat([df3,df_temp],ignore_index=True)
df3['classification'] = 0
df3['classification'] = df3['mid_price_change'].apply(lambda x: 0 if x == 0 else 1 if x > 0 else -1)
df3 = df3.drop(['mid_price_change'],axis=1)
df3


# In[259]:


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
df3 


# In[260]:


# oversampling using Smote algorithm
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
df3

