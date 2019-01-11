# Deep Learning in Finance


The paper we are studying is : "Deep Learning for Spatio-Temporal Modeling: Dynamic Traffic Flows and High Frequency Trading". The latter exposes how deep learning architectures can be used to spatio-temporal predictive modelling. 

The modelling challenge we face is the following : we consider a certain number of snapshots of the limit order book on a stock as predictors, in order to estimate the future movement of the stock’s mid price. It is a classification problem and the output is one of the following class:
- The mid-price is going up 
- The mid-price is going down
- The mid-price is staying the same


We started by implementing two Feed Forward Neural Nets respecting the architecture of the original paper. The first one, is simply a reproduction of the original work. We added, as a direct extension of the paper,to the second architecture the spread of the book and therefore modified the structure of the DNN to incoporate this new information. 

As a more sophisticated approach, we implemented two RNN with LSTM cells. The first is a one layer RNN. The second a multilayered RNN as several papers show the importance of stacking layers.

To try the scripts, simply download them and the order book data. Put the script and data in the same repository, select the number of rows (start with 10000 rows in our opinion) and run the script. The end result will be different graphs showing the predictive power of the trained graph.

For the Neural Turing Machine code, you need to dowload the library from the repo: https://github.com/flomlo/ntm_keras/blob/master/ntm.py?fbclid=IwAR1vmFqxlmP0pNptPTPbekqPKy7QfsXjzqjYIQ32EyXedBgxPgHGSG6kvos
and then dowload the script and import the ntm as a library.

We inspired our work from several papers:

- Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent
neural networks.
- Michiel Hermans and Benjamin Schrauwen. Training and analysing deep recurrent neural networks.
- Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, and Yoshua Bengio. How to construct deep recurrent
neural networks.
- Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu. Recurrent neural
networks for multivariate time series with missing values.
- Matthew F Dixon, Nicholas G Polson, and Vadim O Sokolov. Deep learning for spatio-temporal modeling:
Dynamic traffic flows and high frequency trading.
- Pankaj Malhotra, Lovekesh Vig, Gautam Shroff, and Puneet Agarwal. Long short term memory networks
for anomaly detection in time series. 
