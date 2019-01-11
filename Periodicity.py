# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:16:48 2019

@author: Othman
"""
"""
This is the code for calculating the relatif time that elapses between two consecutive price movement. The plus_num and moins_num will 
reflect the number of times the mid price change under 30, 100, 300 and 1000 steps. It gives a sense of the different periodicity scales
by trying different values of time steps.
"""

import os
import pandas as pd


cwd = os.getcwd()

path='Data for DNN'
os.chdir(path)

cwd = os.getcwd()

## Load csv

file_path = 'ESZ2018.csv'

n_levels=10    
nrows=500000  
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


plus=0
moins=0
for i in range(len(df2)):
    if df2["classification"][i]==1 :
        plus+=1
    elif df2["classification"][i]==-1:
        moins+=1
plus_pos=[]
moins_pos=[]
for i in range(len(df2)):
    if df2["classification"][i]==1 :
        plus_pos.append(i)
    elif df2["classification"][i]==-1:
        moins_pos.append(i)
        
relat_plus_pos=[]
relat_moins_pos=[]        
for i in range(len(plus_pos)-1):
        relat_plus_pos.append(plus_pos[i+1]-plus_pos[i])
   
for i in range(len(moins_pos)-1):
        relat_moins_pos.append(moins_pos[i+1]-moins_pos[i])

moins_num=[0,0,0,0,0]        
for i in range(len(relat_moins_pos)-1):
    if relat_moins_pos[i]<30:
        moins_num[0]+=1
    elif relat_moins_pos[i]<100:
        moins_num[1]+=1
    elif relat_moins_pos[i]<300:
        moins_num[2]+=1
    elif relat_moins_pos[i]<1000:
        moins_num[3]+=1
    else: 
         moins_num[3]+=1
plus_num=[0,0,0,0,0]
for i in range(len(relat_plus_pos)-1):
    if relat_plus_pos[i]<30:
        plus_num[0]+=1
    elif relat_plus_pos[i]<100:
        plus_num[1]+=1
    elif relat_plus_pos[i]<300:
        plus_num[2]+=1
    elif relat_plus_pos[i]<1000:
        plus_num[3]+=1
    else: plus_num[4]+=1