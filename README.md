# Deep Learning in Finance


The paper we are studying is : "Deep Learning for Spatio-Temporal Modeling: Dynamic Traffic Flows and High Frequency Trading". The latter exposes how deep learning architectures can be used to spatio-temporal predictive modelling. 
The modelling challenge we face is the following : we consider a certain number of snapshots of the limit order book on a stock as predictors, in order to estimate the future movement of the stock’s mid price. It is a classification problem and the output is one of the following class:
- The mid-price is going up 
- The mid-price is going down
- The mid-price is staying the same

As complex and often nonlinear interactions exist between the orders’ volumes at different levels andtimes, this modelling problem reveals to be challenging. The deep learning models we develop in this paper apply layers of hierarchical hidden variables to capture these interactions. 
In a first step, we will start our research with a rather simple deep learning architecture : a fully connected feed forward neural network. To train this model, we use limit order book data on "E-MiniSP 500 Futures" for the month of November. The most challenging part in the training of this model is the pre-processing of the data. Indeed, there is a strong imbalance in the observations of the different classes, as the mid price changes very few times in a window where the time step is approximately a millisecond. To solve this problem, we use a method which relies on the combination of undersampling and over-sampling techniques. We use the SMOTE algorithm (Synthetic Minority Oversampling TEchnique).
The second step we take is that we consider a Recurrent Neural Network architecture. The ability of RNN structure to capture the dependency to previously perceived inputs allows it to have a memory, which is the improvement we want for the Feed Forward Network. We focus on the famous LSTM (Long-ShortTerm Memory) which has proven to be extremely efficient in similar problems as ours. First, we train an LSTM structure on the mid-price values through time. Then, we train it on the limit order book snapshots.
