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
from metrics.Metrics import Metrics
from sklearn.model_selection import train_test_split
from utils import *

batch_size = 128
epochs = 10
max_len = 150

features_num_dict = {'click_times': 11, 'click_times.1': 11, 'click_times.2': 10, 'click_times.3': 9, 'click_times.4': 9, 'click_times.5': 9, 'click_times.6': 9, 'click_times.7': 9, 'click_times.8': 9, 'click_times.9': 9, 'click_times.10': 4, 'click_times.11': 11, 'click_times.12': 10, 'click_times.13': 10, 'click_times.14': 9, 'click_times.15': 8, 'click_times.16': 8, 'click_times.17': 9, 'click_times.18': 8, 'click_times.19': 3, 'click_times.20': 5, 'click_times.21': 6, 'click_times.22': 8, 'click_times.23': 3, 'click_times.24': 4, 'click_times.25': 7, 'click_times.26': 10, 'click_times.27': 10, 'click_times.28': 7, 'click_times.29': 6, 'click_times.30': 5, 'click_times.31': 5, 'click_times.32': 7, 'click_times.33': 10, 'click_times.34': 1, 'click_times.35': 8, 'click_times.36': 3, 'click_times.37': 3, 'click_times.38': 4, 'click_times.39': 5, 'click_times.40': 7, 'click_times.41': 2, 'click_times.42': 2, 'click_times.43': 3, 'click_times.44': 2, 'click_times.45': 5, 'click_times.46': 4, 'click_times.47': 6, 'click_times.48': 9, 'click_times.49': 3, 'click_times.50': 6, 'click_times.51': 8, 'click_times.52': 8, 'click_times.53': 7, 'click_times.54': 7, 'click_times.55': 6, 'click_times.56': 7, 'click_times.57': 4, 'click_times.58': 7, 'click_times.59': 4, 'click_times.60': 3, 'click_times.61': 8, 'click_times.62': 6, 'click_times.63': 7, 'click_times.64': 2, 'click_times.65': 5, 'click_times.66': 3, 'click_times.67': 7, 'click_times.68': 2, 'click_times.69': 2, 'click_times.70': 1, 'click_times.71': 5, 'click_times.72': 6, 'click_times.73': 4, 'click_times.74': 8, 'click_times.75': 3, 'click_times.76': 3, 'click_times.77': 3, 'click_times.78': 4, 'click_times.79': 2, 'click_times.80': 5, 'click_times.81': 9, 'click_times.82': 4, 'click_times.83': 2, 'click_times.84': 6, 'click_times.85': 4, 'click_times.86': 2, 'click_times.87': 8, 'click_times.88': 6, 'click_times.89': 4, 'click_times.90': 3, 'click_times.91': 5, 'click_times.92': 4, 'click_times.93': 4, 'click_times.94': 2, 'click_times.95': 2, 'click_times.96': 2, 'click_times.97': 2, 'click_times.98': 2, 'click_times.99': 4, 'click_times.100': 8, 'click_times.101': 6, 'click_times.102': 4, 'click_times.103': 2, 'click_times.104': 1, 'click_times.105': 2, 'click_times.106': 2, 'click_times.107': 1, 'click_times.108': 6, 'click_times.109': 6, 'click_times.110': 2, 'click_times.111': 7, 'click_times.112': 4, 'click_times.113': 6, 'click_times.114': 5, 'click_times.115': 6, 'click_times.116': 6, 'click_times.117': 3, 'click_times.118': 1, 'click_times.119': 2, 'click_times.120': 2, 'click_times.121': 2, 'click_times.122': 1, 'click_times.123': 2, 'click_times.124': 2, 'click_times.125': 5, 'click_times.126': 5, 'click_times.127': 5, 'click_times.128': 2, 'click_times.129': 2, 'click_times.130': 2, 'click_times.131': 6, 'click_times.132': 3, 'click_times.133': 7, 'click_times.134': 8, 'click_times.135': 5, 'click_times.136': 6, 'click_times.137': 3, 'click_times.138': 5, 'click_times.139': 3, 'click_times.140': 5, 'click_times.141': 4, 'click_times.142': 3, 'click_times.143': 6, 'click_times.144': 2, 'click_times.145': 3, 'click_times.146': 3, 'click_times.147': 3, 'click_times.148': 2, 'click_times.149': 3, 'click_times.150': 4, 'click_times.151': 6, 'click_times.152': 3, 'click_times.153': 5, 'click_times.154': 3, 'click_times.155': 3, 'click_times.156': 5, 'click_times.157': 4, 'click_times.158': 2, 'click_times.159': 2, 'click_times.160': 6, 'click_times.161': 8, 'click_times.162': 5, 'click_times.163': 4, 'click_times.164': 5, 'click_times.165': 6, 'click_times.166': 3, 'click_times.167': 2, 'click_times.168': 5, 'click_times.169': 2, 'click_times.170': 2, 'click_times.171': 3, 'click_times.172': 3, 'click_times.173': 3, 'click_times.174': 6, 'click_times.175': 6, 'click_times.176': 5, 'click_times.177': 2, 'click_times.178': 4, 'click_times.179': 2, 'click_times.180': 1, 'click_times.181': 6, 'click_times.182': 4, 'click_times.183': 3, 'click_times.184': 5, 'click_times.185': 5, 'click_times.186': 3, 'click_times.187': 2, 'click_times.188': 7, 'click_times.189': 5, 'click_times.190': 6, 'click_times.191': 3, 'click_times.192': 6, 'click_times.193': 3, 'click_times.194': 8, 'click_times.195': 6, 'click_times.196': 3, 'click_times.197': 4, 'click_times.198': 6, 'click_times.199': 5, 'click_times.200': 6, 'click_times.201': 8, 'click_times.202': 5, 'click_times.203': 7, 'click_times.204': 3, 'click_times.205': 4, 'click_times.206': 6, 'click_times.207': 2, 'click_times.208': 2, 'click_times.209': 6, 'click_times.210': 8, 'click_times.211': 3, 'click_times.212': 3, 'click_times.213': 5, 'click_times.214': 4, 'click_times.215': 7, 'click_times.216': 1, 'click_times.217': 2, 'click_times.218': 2, 'click_times.219': 6, 'click_times.220': 2, 'click_times.221': 2, 'click_times.222': 6, 'click_times.223': 2, 'click_times.224': 2, 'click_times.225': 1, 'click_times.226': 1, 'click_times.227': 8, 'click_times.228': 5, 'click_times.229': 6, 'click_times.230': 6, 'click_times.231': 6, 'click_times.232': 5, 'click_times.233': 4, 'click_times.234': 7, 'click_times.235': 3, 'click_times.236': 6, 'click_times.237': 3, 'click_times.238': 3, 'click_times.239': 2, 'click_times.240': 2, 'click_times.241': 3, 'click_times.242': 6, 'click_times.243': 6, 'click_times.244': 6, 'click_times.245': 3, 'click_times.246': 5, 'click_times.247': 5, 'click_times.248': 5, 'click_times.249': 5, 'click_times.250': 3, 'click_times.251': 3, 'click_times.252': 3, 'click_times.253': 3, 'click_times.254': 4, 'click_times.255': 3, 'click_times.256': 6, 'click_times.257': 7, 'click_times.258': 9, 'click_times.259': 7, 'click_times.260': 7, 'click_times.261': 5, 'click_times.262': 5, 'click_times.263': 5, 'click_times.264': 5, 'click_times.265': 10, 'click_times.266': 3, 'click_times.267': 4, 'click_times.268': 5, 'click_times.269': 9, 'click_times.270': 2, 'click_times.271': 2, 'click_times.272': 6, 'click_times.273': 7, 'click_times.274': 10, 'click_times.275': 8, 'click_times.276': 5, 'click_times.277': 8, 'click_times.278': 5, 'click_times.279': 6, 'click_times.280': 8, 'click_times.281': 6, 'click_times.282': 6, 'click_times.283': 5, 'click_times.284': 4, 'click_times.285': 5, 'click_times.286': 6, 'click_times.287': 6, 'click_times.288': 6, 'click_times.289': 6, 'click_times.290': 3, 'click_times.291': 3, 'click_times.292': 3, 'click_times.293': 2, 'click_times.294': 2, 'click_times.295': 2, 'click_times.296': 2, 'click_times.297': 3, 'click_times.298': 5, 'click_times.299': 3, 'click_times.300': 2, 'click_times.301': 2, 'click_times.302': 4, 'click_times.303': 2, 'click_times.304': 6, 'click_times.305': 2, 'click_times.306': 2, 'click_times.307': 2, 'click_times.308': 4, 'click_times.309': 2, 'click_times.310': 2, 'click_times.311': 3, 'click_times.312': 3, 'click_times.313': 5, 'click_times.314': 2, 'click_times.315': 6, 'click_times.316': 7, 'click_times.317': 6, 'click_times.318': 6, 'click_times.319': 5, 'click_times.320': 6, 'click_times.321': 4, 'click_times.322': 5, 'click_times.323': 8, 'click_times.324': 6, 'click_times.325': 4, 'click_times.326': 3, 'click_times.327': 8, 'click_times.328': 6, 'click_times.329': 5, 'click_times.330': 4, 'click_times.331': 3, 'click_times.332': 6, 'click_times.333': 3, 'click_times.334': 4, 'click_times.335': 4, 'click_times.336': 3, 'click_times.337': 3, 'click_times.338': 3, 'click_times.339': 4, 'click_times.340': 3, 'click_times.341': 3, 'click_times.342': 3, 'click_times.343': 4, 'click_times.344': 8, 'click_times.345': 8, 'click_times.346': 9, 'click_times.347': 7, 'click_times.348': 8, 'click_times.349': 10, 'click_times.350': 6, 'click_times.351': 3, 'click_times.352': 2, 'click_times.353': 9, 'click_times.354': 5, 'click_times.355': 6, 'click_times.356': 8, 'click_times.357': 5, 'click_times.358': 3, 'click_times.359': 3, 'click_times.360': 5, 'click_times.361': 5, 'click_times.362': 5, \
    'time_len':max_len,'time_size':92,'creative_id_len':max_len,'creative_id_size':26713,'ad_id_len':max_len,'ad_id_size':26713,'advertiser_id_len':max_len,'advertiser_id_size':26713,'industry_len':max_len,'industry_size':336,'product_id_len':max_len,'product_id_size':26713,'product_category_len':max_len,'product_category_size':19,}

sparse_features = ['click_times','click_times.1','click_times.2','click_times.3','click_times.4','click_times.5','click_times.6','click_times.7','click_times.8','click_times.9','click_times.10','click_times.11','click_times.12','click_times.13','click_times.14','click_times.15','click_times.16','click_times.17','click_times.18','click_times.19','click_times.20','click_times.21','click_times.22','click_times.23','click_times.24','click_times.25','click_times.26','click_times.27','click_times.28','click_times.29','click_times.30','click_times.31','click_times.32','click_times.33','click_times.34','click_times.35','click_times.36','click_times.37','click_times.38','click_times.39','click_times.40','click_times.41','click_times.42','click_times.43','click_times.44','click_times.45','click_times.46','click_times.47','click_times.48','click_times.49','click_times.50','click_times.51','click_times.52','click_times.53','click_times.54','click_times.55','click_times.56','click_times.57','click_times.58','click_times.59','click_times.60','click_times.61','click_times.62','click_times.63','click_times.64','click_times.65','click_times.66','click_times.67','click_times.68','click_times.69','click_times.70','click_times.71','click_times.72','click_times.73','click_times.74','click_times.75','click_times.76','click_times.77','click_times.78','click_times.79','click_times.80','click_times.81','click_times.82','click_times.83','click_times.84','click_times.85','click_times.86','click_times.87','click_times.88','click_times.89','click_times.90','click_times.91','click_times.92','click_times.93','click_times.94','click_times.95','click_times.96','click_times.97','click_times.98','click_times.99','click_times.100','click_times.101','click_times.102','click_times.103','click_times.104','click_times.105','click_times.106','click_times.107','click_times.108','click_times.109','click_times.110','click_times.111','click_times.112','click_times.113','click_times.114','click_times.115','click_times.116','click_times.117','click_times.118','click_times.119','click_times.120','click_times.121','click_times.122','click_times.123','click_times.124','click_times.125','click_times.126','click_times.127','click_times.128','click_times.129','click_times.130','click_times.131','click_times.132','click_times.133','click_times.134','click_times.135','click_times.136','click_times.137','click_times.138','click_times.139','click_times.140','click_times.141','click_times.142','click_times.143','click_times.144','click_times.145','click_times.146','click_times.147','click_times.148','click_times.149','click_times.150','click_times.151','click_times.152','click_times.153','click_times.154','click_times.155','click_times.156','click_times.157','click_times.158','click_times.159','click_times.160','click_times.161','click_times.162','click_times.163','click_times.164','click_times.165','click_times.166','click_times.167','click_times.168','click_times.169','click_times.170','click_times.171','click_times.172','click_times.173','click_times.174','click_times.175','click_times.176','click_times.177','click_times.178','click_times.179','click_times.180','click_times.181','click_times.182','click_times.183','click_times.184','click_times.185','click_times.186','click_times.187','click_times.188','click_times.189','click_times.190','click_times.191','click_times.192','click_times.193','click_times.194','click_times.195','click_times.196','click_times.197','click_times.198','click_times.199','click_times.200','click_times.201','click_times.202','click_times.203','click_times.204','click_times.205','click_times.206','click_times.207','click_times.208','click_times.209','click_times.210','click_times.211','click_times.212','click_times.213','click_times.214','click_times.215','click_times.216','click_times.217','click_times.218','click_times.219','click_times.220','click_times.221','click_times.222','click_times.223','click_times.224','click_times.225','click_times.226','click_times.227','click_times.228','click_times.229','click_times.230','click_times.231','click_times.232','click_times.233','click_times.234','click_times.235','click_times.236','click_times.237','click_times.238','click_times.239','click_times.240','click_times.241','click_times.242','click_times.243','click_times.244','click_times.245','click_times.246','click_times.247','click_times.248','click_times.249','click_times.250','click_times.251','click_times.252','click_times.253','click_times.254','click_times.255','click_times.256','click_times.257','click_times.258','click_times.259','click_times.260','click_times.261','click_times.262','click_times.263','click_times.264','click_times.265','click_times.266','click_times.267','click_times.268','click_times.269','click_times.270','click_times.271','click_times.272','click_times.273','click_times.274','click_times.275','click_times.276','click_times.277','click_times.278','click_times.279','click_times.280','click_times.281','click_times.282','click_times.283','click_times.284','click_times.285','click_times.286','click_times.287','click_times.288','click_times.289','click_times.290','click_times.291','click_times.292','click_times.293','click_times.294','click_times.295','click_times.296','click_times.297','click_times.298','click_times.299','click_times.300','click_times.301','click_times.302','click_times.303','click_times.304','click_times.305','click_times.306','click_times.307','click_times.308','click_times.309','click_times.310','click_times.311','click_times.312','click_times.313','click_times.314','click_times.315','click_times.316','click_times.317','click_times.318','click_times.319','click_times.320','click_times.321','click_times.322','click_times.323','click_times.324','click_times.325','click_times.326','click_times.327','click_times.328','click_times.329','click_times.330','click_times.331','click_times.332','click_times.333','click_times.334','click_times.335','click_times.336','click_times.337','click_times.338','click_times.339','click_times.340','click_times.341','click_times.342','click_times.343','click_times.344','click_times.345','click_times.346','click_times.347','click_times.348','click_times.349','click_times.350','click_times.351','click_times.352','click_times.353','click_times.354','click_times.355','click_times.356','click_times.357','click_times.358','click_times.359','click_times.360','click_times.361','click_times.362']

# list_features = ['time', 'creative_id', 'ad_id', 'advertiser_id', 'industry','product_id', 'product_category']
list_features = [ 'advertiser_id','product_id']

project = '/home/tione/notebook/'

user_path = project + 'train/user/user.csv'
user_cwwpi_path = project + 'train/user/user_cwwpi.csv'
user_sequence_path = project + 'train/user/user_sequence.csv'

## 0 label
label = 'gender'

## 1. load raw data
user = pd.read_csv(user_path)
user_cwwpi = pd.read_csv(user_cwwpi_path)
user_sequence = pd.read_csv(user_sequence_path)

user = user[user['user_id']!=839368]

raw_df = pd.merge(user,user_cwwpi,on='user_id')
del user
del user_cwwpi
raw_df = pd.merge(raw_df,user_sequence,on='user_id')
del user_sequence
print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :raw_df made")

## 2.split train test
X,y = raw_df[sparse_features+list_features],raw_df[label]-1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1024)
del X
del y
del raw_df
print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :  train test splited")


## 3.1 process train  data
model_input = {name: X_train[name].fillna(0).map(lambda x:int(math.log(x+1,2))).astype(np.int16) for name in sparse_features}
X_train.drop(sparse_features,axis=1,inplace=True)
print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :X_train sparse processed")

for feature in list_features:
    if feature in ['product_id','industry']:
        feature_list =  list(map(eval_with_nan, X_train[feature].values))
    else:
        feature_list =  list(map(myeval, X_train[feature].values))
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
    if feature in ['product_id','industry']:
        feature_list =  list(map(eval_with_nan, X_test[feature].values))
    else:
        feature_list =  list(map(myeval, X_test[feature].values))
    X_test.drop([feature],axis=1,inplace=True)
    print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :X_test "+ feature +" dropped")
    feature_list = pad_sequences(feature_list, maxlen=features_num_dict[feature+"_len"],dtype='int16')
    model_output[feature] = feature_list
    del feature_list
del X_test
print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :X_test list processed")


## 4. Define Model, train, predict and evaluate

checkpoint = ModelCheckpoint('models/gender_v1.h5', save_weights_only=False, save_best_only=True)
callbacks_list = [checkpoint,EarlyStopping(monitor='val_acc',patience=0)]


model = DeepFM(sparse_features, list_features,features_num_dict,k=10,list_k=100).model
model.compile("adam", "binary_crossentropy",metrics=['binary_crossentropy','acc'],)

history = model.fit(model_input, y_train,batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,validation_data=(model_output,y_test),callbacks=callbacks_list)

model.save('models/gender_v1_final.h5')
