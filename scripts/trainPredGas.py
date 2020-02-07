#!/usr/bin/env python
# coding: utf-8

# In[126]:


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


# In[4]:


#add code to input data
#for now, uploads data file from local drive
dataset1 =  pd.read_csv('cinTargetClean.csv')
dataset2 =  pd.read_csv('cleTargetClean.csv')


# In[5]:


#combine cleveland and cinncinati target datasets into one
dataset = pd.concat([dataset1, dataset2])


# In[15]:


#drop client name column
dataset.drop('FullName', axis = 1, inplace = True)


# In[8]:


#deleting the init. datasets now that they're one large dataset
del dataset1
del dataset2


# In[39]:


#set home value at/above 100k
dataset = dataset[dataset['Value'] >= 100000.00].copy()


# In[77]:


#set energy value above 6000 KwH Annual
dataset = dataset[dataset['E annual'] >= 6000.00].copy()


# In[101]:


#set energy value less than 6000 KwH Annual
dataset = dataset[dataset['G annual'] <= 4000.00].copy()


# In[102]:


#check to ensure dataset concat. correctly
#uncomment to run
#dataset.shape


# In[105]:


train_DF = dataset.describe()
train_DF = train_DF.transpose()
train_DF


# In[106]:


#sample dataset (10% of full data) to test parameter tuning with
#uncomment to run
#dataset = dataset.sample(frac=0.1, random_state=0)


# In[122]:


#split data into train/test sets 80%/20%
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# In[127]:


#creates training data stats 
train_stats = train_dataset.describe()
train_stats.pop("G annual")
train_stats = train_stats.transpose()

#save train stats for norm. function
scaler_filename = "trainPredGasScaler.save"
joblib.dump(train_stats, scaler_filename)


# In[110]:


#train_labels


# In[111]:


#creates train and test datasets target variable: Annual Natural Gas Usage
#and removes target variable from feature variable dataset
train_labels = train_dataset.pop('G annual')
test_labels = test_dataset.pop('G annual')


# In[112]:


#Function to normalize train and test datasets to dist. range from train data
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


# In[113]:


#normalized train and test datasets
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# In[114]:


#Deep Learning model, 8 layers deep, 2 fully connected layers
#can adjust optimizer by commenting/uncommeting 'optimizer' variable
#loss func. set to highly penalize model for large errors

def build_model():
  model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=[len(train_dataset.keys())]),
    #layers.Dropout(0.25),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.15),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.15),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.15),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.15),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.15),
    #layers.Dense(8, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  #optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(loss='mean_squared_logarithmic_error',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


# In[115]:


#calls ML model and assigns it to the variable 'model'
model = build_model()


# In[116]:


# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)

model.fit(normed_train_data, train_labels, epochs=1000, 
          validation_split = 0.2, verbose=2, 
          callbacks=[early_stop])


# In[117]:


#check models loss, mean actual error rate, mean squared error rate
#uncomment to run
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} ".format(mae))


# In[118]:


#saves TF model as name noted below
model.save('predAnnualGas.h5')


# In[119]:


#Save the model, serialized model to JSON format
model_json = model.to_json()
with open("predGasModel.json", "w") as json_file:
    json_file.write(model_json)

