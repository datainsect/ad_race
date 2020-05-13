import os
import sys
import time
import numpy as np
import pandas as pd
import keras
import math
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Reshape, Lambda, concatenate, dot, add
from keras.layers import Dropout, GaussianDropout, multiply, SpatialDropout1D, BatchNormalization, subtract
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.framework import graph_util
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam


import time
from keras.preprocessing.sequence import pad_sequences

from dnn.DeepFM import DeepFM
from dnn.DeepFM1 import DeepFM1
from metrics.Metrics import Metrics
from sklearn.model_selection import train_test_split

from const import *
batch_size = 32
epochs = 5
max_len = 150



def myeval(s):
    l = eval(s)
    if len(l)==0:
        l.append(1)
    return l

## 

label = 'age'

list_features = [ 'advertiser_id']

## 1. load raw data
df = pd.read_csv(raw_processed)

df = df[df.total_times<=335]

df = df[df.user_id<=900000]

user = pd.read_csv(train_user_path)

sparse_features = list(df.columns[1:-3])

new_df = df[['user_id']+sparse_features+list_features]

raw_df = pd.merge(new_df,user,on='user_id')

## 2.split train test

X,y = raw_df[sparse_features+list_features],raw_df[label]-1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1024)
del X
del y
del raw_df
print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :  train test splited")


## 3.1 process train  data
model_input = {name: X_train[name].fillna(0).map(lambda x:int(math.log(x+1,2))).astype(np.int16) for name in sparse_features}

X_train.drop(sparse_features,axis=1,inplace=True)
print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :X_train sparse processed")

for feature in list_features:
    feature_list =  list(map(myeval, map(str,X_train[feature].values)))
    X_train.drop([feature],axis=1,inplace=True)
    print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :X_train "+ feature +" dropped")
    feature_list = pad_sequences(feature_list, maxlen=features_num_dict[feature+"_len"],dtype='int16')
    model_input[feature] = feature_list
    del feature_list

del X_train
print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :X_train list processed")

## 3.2 process test  data
model_output = {name: X_test[name].fillna(0).map(lambda x:int(math.log(x+1,2))).astype(np.int16) for name in sparse_features}
X_test.drop(sparse_features,axis=1,inplace=True)
print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :X_test sparse processed")

for feature in list_features:
    feature_list =  list(map(myeval, map(str,X_test[feature].values)))
    X_test.drop([feature],axis=1,inplace=True)
    print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :X_test "+ feature +" dropped")
    feature_list = pad_sequences(feature_list, maxlen=features_num_dict[feature+"_len"],dtype='int16')
    model_output[feature] = feature_list
    del feature_list


del X_test
print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :X_test list processed")


## 4. Define Model, train, predict and evaluate

checkpoint = ModelCheckpoint('models/deepfm4.h5', save_weights_only=False, verbose=1, save_best_only=True)
callbacks_list = [checkpoint] 

model = DeepFM(sparse_features, list_features,features_num_dict,k=10,list_k=50).model
model.compile("adam", "binary_crossentropy",metrics=['binary_crossentropy','acc'],)

history = model.fit(model_input, y_train,batch_size=batch_size, epochs=epochs, verbose=2, shuffle=True,validation_data=(model_output,y_test),callbacks=callbacks_list)

model.save('models/deepfm4_final.h5')