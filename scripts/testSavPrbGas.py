#!/usr/bin/env python
# coding: utf-8

# In[1]:


#packages
#python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
#packages
import pathlib
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import model_from_json


# In[2]:


#load gas model
json_file = open('predGasModel.json','r')
loaded_model_jsonGAS = json_file.read()
json_file.close()
loaded_modelGAS = model_from_json(loaded_model_jsonGAS)

#load weights into new model
loaded_modelGAS.load_weights("predAnnualGas.h5")
print("Loaded Gas Model from disk")


# In[ ]:


#load gas model scaler weights
scalerGas = joblib.load("trainPredGasScaler.save") 


# In[3]:


#load pred. savings model
json_file = open('predSaveModel.json','r')
loaded_model_jsonPrdSave = json_file.read()
json_file.close()
loaded_modelPrdSave = model_from_json(loaded_model_jsonPrdSave)

#load weights into new model
loaded_modelPrdSave.load_weights("predSaveDollar.h5")
print("Loaded Savings Model from disk")


# In[ ]:


#load savings model scaler weights
scalerSaveRate = joblib.load("trainPredSaveScaler.save") 


# In[ ]:


#Function to normalize test data value to dist. range from train data
def norm(x, scaler):
    return (x - scaler['mean']) / scaler['std']


# In[4]:


#load test GAS data
#define how to input the values here
testGas = 'some input type here'


# In[ ]:


#normalize test gas data
testGasNorm = norm(testGas, scalerGas)


# In[ ]:


#compile loaded GAS model
optimizerGAS = tf.keras.optimizers.RMSprop(0.001)
loaded_modelGAS.compile(loss='mean_squared_logarithmic_error', optimizer=optimizerGAS, metrics=['mae', 'mse'])


# In[5]:


#get gas prediction
outputGasPredVal = loaded_modelGAS.predict(testGasNorm)
print(outputGasPredVal)


# In[ ]:


#load test SAVINGS data
#define how to input the values here
testSaveRate = 'some input type here'


# In[6]:


#add gas prediction to savings data
testSaveRate['G annual']  = outputGasPredVal.round()


# In[ ]:


#normalize test savings data
testSaveRateNorm = norm(testSaveRate, scalerSaveRate)


# In[ ]:


#compile loaded SAVINGS model
optimizerSAVE = tf.keras.optimizers.RMSprop(0.001)
loaded_modelSAVE.compile(loss='mean_squared_logarithmic_error', optimizer=optimizerSAVE, metrics=['mae', 'mse'])


# In[7]:


#pass data to savings model, get prediction
outputSavePredVal = loaded_modelPrdSave.predict(testSaveRateNorm)
print(outputSavePredVal)


# In[8]:


#elec data x local value
elecMltprVal = 0.13


# In[9]:


#gas data x local value
gasMltprVal = 8.18


# In[10]:


#combine values to calc. annual dollar elec. and gas cost contd..
#est. annual total utility cost before and after savings and savings percentage
#returns savings rate as pct
#calc annual energy dollar val for central Ohio
energyLocYrCost = testSaveRateOutput['E annual'] * elecMltprVal
#calc annual gas dollar val for central Ohio
gasLocYrCost = testSaveRateOutput['G annual'] * gasMltprVal
#calc est. annual combined cost of gas/elec. utilities
totActUtil = int(energyLocYrCost + gasLocYrCost)
#est new annual utility cost less pred. savings value
newEstUtil = int(totActUtil - outputSavePredVal)
#converts savings value to percentage value
estPctSavings = int((outputSavePredVal/totActUtil) * 100)


# In[ ]:


#return all new prediction values as list that can be parsed
#output format: [current annual util. dollar value, new annual util. dollar value, est. annual savings dollar amount, est. savings rate as percentage]
outputDataVal = [totActUtil, newEstUtil, outputSavePredVal, estPctSavings]

#for demo purposes, prints values below, uncomment to run
#print("Current Annual Utility Cost: %d" % (outputDataVal[0]))
#print("Estimated New Annual Utility Cost: %d" % (outputDataVal[1]))
#print("Estimated Annual Savings Amount: $ %d" % (outputDataVal[2]))
#print("Estimated Annual Savings Percent: %d" % (outputDataVal[3]))


# In[ ]:


#define how you want to push output data to be used/returned to user

