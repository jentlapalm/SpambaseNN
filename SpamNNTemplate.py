# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:03:29 2018

@author: GEASTON
"""
#%% import packages, set some parameters, and get the data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

NEpochs = 10000
BatchSize=250
Optimizer=optimizers.RMSprop(lr=0.001)

# Read in the data

TrainDF = pd.read_csv('NNHWTrain.csv',sep=',',header=0,quotechar='"')
list(TrainDF)
ValDF = pd.read_csv('NNHWVal.csv',sep=',',header=0,quotechar='"')
list(ValDF)
TestDF = pd.read_csv('NNHWTest.csv',sep=',',header=0,quotechar='"')
list(TestDF)

#

TrIsSpam = np.array(TrainDF['IsSpam'])

TrX = np.array(TrainDF.iloc[:,:-1])

# **** Your code to rescale the training X data goes here

# No need to rescale the Y because it is already 0 and 1. But check
print(TrIsSpam.min())
print(TrIsSpam.max())

# Rescale the validation data

ValIsSpam = np.array(ValDF['IsSpam'])

ValX = np.array(ValDF.iloc[:,:-1])

# **** Your code to rescale the test X data goes here

# Rescale the test data

TestIsSpam = np.array(TestDF['IsSpam'])

TestX = np.array(TestDF.iloc[:,:-1])

# **** Your code to rescale the test X data goes here

#%% Set up Neural Net Model

# **** Your code to set up and compile the neural net model goes here

#%% Fit NN Model

from keras.callbacks import EarlyStopping

# **** Your code to fit the neural net model goes here

#%% Make Predictions

# **** Your code to compute the predicted probabilities goes here.
# Do not change the variable names or the code in the next block will not work.

TrP = 
ValP = 
TestP = 

#%% Write out prediction

TrainDF['TrP'] = TrP.reshape(-1)
ValDF['ValP'] = ValP.reshape(-1)
TestDF['TestP'] = TestP.reshape(-1)

TrainDF.to_csv('SpamNNWideTrainDFOutput.csv',sep=',',na_rep="NA",header=True,index=False)
ValDF.to_csv('SpamNNWideValDFOutput.csv',sep=',',na_rep="NA",header=True,index=False)
TestDF.to_csv('SpamNNWideTestDFOutput.csv',sep=',',na_rep="NA",header=True,index=False)

