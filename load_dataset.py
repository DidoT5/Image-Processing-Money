# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 20:31:24 2021

@author: Enrique
"""
from funciones_auxiliares import *
import numpy as np

res = 128

def cargaDataset(carpetas,formato,detras=False):
    X_img = []
    Y = []
    num_clases = len(carpetas)
    clases = ['10cent', '1euro', '20cent', '2euro', '50cent', '5cent']
    cod_clases = [1,4,2,5,3,0]
    if detras:
        cod_clases = [7,10,8,11,9,6]
    for clase,i in zip(clases,cod_clases):
        carpeta=None
        for folder in carpetas:
            if (clase in folder):
                carpeta = folder
                break
        imgs = cargaRuta(carpeta+'\\*{}'.format(formato))
        X_img = X_img + imgs
        Y = Y + [i for j in range(len(imgs))]
    return np.asarray(X_img),np.asarray(Y)

def cargaRuta(ruta):
    monedas = []
    files = glob.glob(ruta)
    for file in files:
        coin = cargaImagen(file)
        coin = cv.resize(coin, (res,res))
        monedas.append(coin)
        
    return monedas

def load_dataset(ruta='./dataset/*',formato=".jpeg",gray=False,detras=False):
    carpetas = glob.glob(ruta)
    X,Y= cargaDataset(carpetas,formato,detras)
    
    #shuffle
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    
    if (gray):
        X = np.asarray([cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in X])

    
    return X,Y

def load_coinsDataset(formato=".jpeg"):
    frontal = './euroDataset/frontal/*';
    frontalTest = './euroDataset/frontal - test/*';
    reverso = './euroDataset/reverso/*';
    reversoTest = './euroDataset/reverso - test/*';
    
    X_train_frontal,Y_train_frontal = cargaDataset(glob.glob(frontal),formato)
    X_test_frontal,Y_test_frontal = cargaDataset(glob.glob(frontalTest),formato)
    X_train_reverso,Y_train_reverso = cargaDataset(glob.glob(reverso),formato,detras=True)
    X_test_reverso,Y_test_reverso = cargaDataset(glob.glob(reversoTest),formato,detras=True)
    
    X_train = np.concatenate((X_train_frontal,X_train_reverso),axis=0)
    Y_train = np.concatenate((Y_train_frontal,Y_train_reverso),axis=0)
    X_test = np.concatenate((X_test_frontal,X_test_reverso),axis=0)
    Y_test = np.concatenate((Y_test_frontal,Y_test_reverso),axis=0)
    
    #shuffle
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    
    indices = np.arange(X_test.shape[0])
    np.random.shuffle(indices)
    X_test = X_test[indices]
    Y_test = Y_test[indices]
    
    return X_train,Y_train,X_test,Y_test
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    