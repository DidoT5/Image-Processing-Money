# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:07:42 2022

@author: Enrique
"""

import cv2 as cv
import numpy as np

from load_dataset import *

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Softmax, Dropout
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical

class detectorMonedas():

    def loadModel(self,pesos):
        input_image = Input(shape=(128,128,3))
        res_model = ResNet50(include_top=False,input_tensor=input_image)
        for layer in res_model.layers[:143]:
            layer.trainable=False
        model = Sequential()
        model.add(res_model)
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(12,activation='softmax'))
        model.load_weights(pesos)
        return model
    
    def __init__(self, pesos="model.hdf5"):
        self.res = 128       
        self.model = self.loadModel(pesos)
        self.valor = [0.10,1,0.20,2,0.50,0.05,0.10,1,0.20,2,0.50,0.05]
        self.correspondencia = [1,4,2,5,3,0,7,10,8,11,9,6]
    
    def preprocess_coins(self,X):
        
        def fondoNegro(moneda):
            xc,yc,radius = self.res//2,self.res//2,self.res//2
            mask = np.zeros((self.res,self.res,3))
            circle = cv.circle(mask, (xc,yc), radius, (255,255,255), -1)
        
            moneda[mask==0] = 0
            return moneda
    
        X = np.asarray([fondoNegro(cv.resize(img,(128,128))) for img in X]).astype('uint8')
        return preprocess_input(X)
    
    
    
    def predecir(self,X):
        processed = self.preprocess_coins(X)
        logits = self.model.predict(processed)
        results = np.argmax(logits,axis=1)
        print(results)
        valores = [self.valor[self.correspondencia.index(result)] for result in results]
        return valores