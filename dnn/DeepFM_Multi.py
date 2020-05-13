from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import os
import time
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Reshape, Lambda, concatenate, dot, add, Masking
from keras.layers import Dropout, GaussianDropout, multiply, SpatialDropout1D, BatchNormalization, subtract
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

sparse_embedding_size=11

class MyMeanPool(Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(MyMeanPool, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.repeat(mask, x.shape[-1])
            mask = tf.transpose(mask, [0,2,1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)
        else:
            return K.mean(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i!=self.axis:
                output_shape.append(input_shape[i])
        return tuple(output_shape)


class DeepFM:
    def __init__(self,cate_features,list_features,features_num_dict,k=10,list_k =10*20, optimizer=Adam(0.001)):
        self.cate_features = cate_features
        self.list_features = list_features
        self.features_num_dict = features_num_dict
        self.k=k
        self.list_k = list_k
        self.optimizer = optimizer
        self.model = self.create_model()
        self.model.summary()

    def create_model(self):
        inputs = []
        field_embedding = []
        list_embedding = []
        ####dealing with sparse input
        for feature in self.cate_features:
            # print('{0} : {1}'.format(feature,self.features_num_dict[feature]))
            input = Input(shape=(1,),name=feature)
            inputs.append(input)
            # fm embeddings
            embed = Embedding(sparse_embedding_size, self.k, input_length=1, trainable=True)(input)
            reshape = Reshape((self.k,))(embed)
            field_embedding.append(reshape)
        ####dealing with sparse list input 
        for features in self.list_features:
            length = self.features_num_dict[features+"_len"]
            input = Input(shape=(length,),name=features)
            inputs.append(input)
            mA = Masking(mask_value=0.0)(input)   
            embed = Embedding(self.features_num_dict[features+"_size"], self.list_k, input_length=length, trainable=True,mask_zero=True)(mA)
            # x = Embedding(output_dim=self.k, input_dim=self.features_num_dict[features+"_size"], input_length=length,mask_zero=True)(input)
            meanpool = MyMeanPool(axis=1)(embed)
            # auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
            list_embedding.append(meanpool)
        #######dnn layer##########
        embed_layer = concatenate(field_embedding+list_embedding, axis=-1)
        embed_layer = Dense(256)(embed_layer)
        # embed_layer = BatchNormalization()(embed_layer)
        embed_layer = Dropout(0.5)(embed_layer)
        embed_layer = Activation('relu')(embed_layer)
        embed_layer = Dense(128)(embed_layer)
        # embed_layer = BatchNormalization()(embed_layer)
        embed_layer = Dropout(0.5)(embed_layer)
        embed_layer = Activation('relu')(embed_layer)
        embed_layer = Dense(64)(embed_layer)
        # embed_layer = BatchNormalization()(embed_layer)
        embed_layer = Dropout(0.5)(embed_layer)
        embed_layer = Activation('relu')(embed_layer)
        ########linear layer##########
        lr_layer = Dense(10)(embed_layer)
        preds = Activation('softmax',name='y_predict')(lr_layer)
        model = Model(inputs=inputs, outputs=preds)
        return model