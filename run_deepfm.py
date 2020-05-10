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

batch_size = 8192
epochs = 5
label = 'is_click'

columns = ['is_click','dt','net','h_channel','model','appver','hour','province','city','gender','age','app_7', 'packages', 
            'cate_list', 'imp_all_28','ctr_all_28',
            'imp_all_14','ctr_all_14','imp_all_7','ctr_all_7','imp_ad_28','ctr_ad_28','dr_ad_28','imp_ad_14','ctr_ad_14','dr_ad_14',
            'imp_ad_7','ctr_ad_7','dr_ad_7','crid','cid','gid','aid','resolution','industry','sub_industry','is_video','is_app','crid_imp_28','crid_cr_28',
            'crid_dr_28','crid_imp_14','crid_cr_14','crid_dr_14','crid_imp_7','crid_cr_7','crid_dr_7']


features_num_dict = {'dt':2,'net':3,'h_channel':28,'model':7,'appver':150,'hour':24,'province':37,'city':361,'gender':3,'age':36,
    'imp_all_28':19,'ctr_all_28':100,'imp_all_14':19,'ctr_all_14':100,'imp_all_7':19,'ctr_all_7':100,'imp_ad_28':19,'ctr_ad_28':100,
    'dr_ad_28':100,'imp_ad_14':19,'ctr_ad_14':100,'dr_ad_14':100,'imp_ad_7':19,'ctr_ad_7':100,'dr_ad_7':100,'crid':7779,'cid':3457,'gid':1235,
    'aid':568,'resolution':185,'industry':13,'sub_industry':30,'is_video':2,'is_app':2,'crid_imp_28':36,'crid_cr_28':100,'crid_dr_28':100,'crid_imp_14':36,
    'crid_cr_14':100,'crid_dr_14':100,'crid_imp_7':36,'crid_cr_7':100,'crid_dr_7':100,
    'click_crid':201,'package_crid':201,'class_name1':428,'class_name2':428,'class_name3':209,
    'app_7_len':101,'app_7_size':7779,'packages_len':101,'packages_size':8200,'cate_list_len':101,'cate_list_size':101,}

sparse_features = ['dt','net','h_channel','model','appver','hour','province','city','gender','age', 'imp_all_28','ctr_all_28',
            'imp_all_14','ctr_all_14','imp_all_7','ctr_all_7','imp_ad_28','ctr_ad_28','dr_ad_28','imp_ad_14','ctr_ad_14','dr_ad_14',
            'imp_ad_7','ctr_ad_7','dr_ad_7','crid','cid','gid','aid','resolution','industry','sub_industry','is_video','is_app','crid_imp_28','crid_cr_28',
            'crid_dr_28','crid_imp_14','crid_cr_14','crid_dr_14','crid_imp_7','crid_cr_7','crid_dr_7']

list_features = ['time', 'creative_id', 'ad_id', 'advertiser_id', 'industry',
       'product_id', 'product_category']

max_len=101


    ## 1. sample train data    
    train = pd.read_csv(train_file,header=None,names=columns)
    train_positive = train[train['is_click']==1]
    train_negtive = train[train['is_click']==0]
    train_negtive = train_negtive.sample(frac=0.5,replace=False)
    train = pd.concat([train_positive,train_negtive])
    del train_negtive
    del train_positive
    train = train[train['dt']==0]
    print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :train sampled")

    ## 2. process train data
    model_input = {name: train[name].fillna(0).astype(np.int16) for name in sparse_features}  
    train.drop(sparse_features,axis=1,inplace=True)
    print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :train sparse processed")
    y_train = train[label].values
    train.drop([label],axis=1,inplace=True)
    for feature in list_features:
        feature_list =  list(map(lambda x :split(x,default=features_num_dict[feature+"_size"]-1), train[feature].values))
        train.drop([feature],axis=1,inplace=True)
        print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :train "+ feature +" dropped")
        feature_list = pad_sequences(feature_list, maxlen=features_num_dict[feature+"_len"],dtype='int16')
        model_input[feature] = feature_list
        # del feature_list
    del train
    print(time.strftime('%Y-%m-%d %H:%M:%S')+ "  :train list processed")
    
    ## 3. Define Model,train,predict and evaluate
    checkpoint = ModelCheckpoint(h5_path, save_weights_only=False, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint] 

    model = DeepFM(sparse_features, list_features,features_num_dict).model
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'],)
    sess=tf.Session()
    K.set_session(sess)
    history = model.fit(model_input, y_train,
                        batch_size=batch_size, epochs=epochs, verbose=2, shuffle=True)
    print(history)

    ## 4. Save model
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=["y_predict/Sigmoid"])
    builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
    builder.add_meta_graph_and_variables(sess,['tecent_race_2020'])
    builder.save()
    sess.close()