#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#add code to input data
#for now, uploads data file from local drive
dataset =  pd.read_csv('mergeCleCinSave.csv')


# In[3]:


#split data into train/test sets 80%/20%
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# In[4]:


#creates training data stats 
train_stats = train_dataset.describe()
train_stats.pop("Estimated Savings")
train_stats = train_stats.transpose()

#save train stats for norm. function
scaler_filename = "trainPredSaveScaler.save"
joblib.dump(train_stats, scaler_filename)


# In[5]:


#creates train and test datasets target variable: Estimated Savings dollar value
#and removes target variable from feature variable dataset
train_labels = train_dataset.pop('Estimated Savings')
test_labels = test_dataset.pop('Estimated Savings')


# In[6]:


#Function to normalize train and test datasets to dist. range from train data
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


# In[7]:


#normalized train and test datasets
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# In[9]:


normed_test_data.head(1)


# In[10]:


#Deep Learning model, 4 layers deep, 2 fully connected layers
#can adjust optimizer by commenting/uncommeting 'optimizer' variable
#loss func. set to mod. penalize model for large errors

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dropout(0.5), #50% data random dropout to reduce overfitting training data
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5), #50% data random dropout to reduce overfitting training data
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  #optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(loss='mean_squared_logarithmic_error',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


# In[11]:


#calls ML model and assigns it to the variable 'model'
model = build_model()


# In[12]:


# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

model.fit(normed_train_data, train_labels, epochs=1000, 
          validation_split = 0.2, verbose=0, 
          callbacks=[early_stop])


# In[13]:


#check models loss, mean actual error rate, mean squared error rate
#uncomment to run
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} ".format(mae))


# In[14]:


#saves TF model weights as name noted below
model.save('predSaveDollar.h5')


# In[15]:


#save the model, serialized model to JSON format
model_json = model.to_json()
with open("predSaveModel.json", "w") as json_file:
    json_file.write(model_json)

