# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 12:03:56 2021

@author: Administrator
"""
import tensorflow as tf
import h5py
import numpy as np

##########################################################################################################

def import_data(every=False):
    if every:
        electrodes = 25
    else:
        electrodes = 12
    X, y = [], []
    for i in range(9):
        A01T = h5py.File('A0' + str(i + 1) + 'T_slice.mat', 'r')
        X1 = np.copy(A01T['image'])
        X.append(X1[:, :electrodes, :])
        y1 = np.copy(A01T['type'])
        y1 = y1[0, 0:X1.shape[0]:1]
        y.append(np.asarray(y1, dtype=np.int32))

    for subject in range(9):
        delete_list = []
        for trial in range(288):
            if np.isnan(X[subject][trial, :, :]).sum() > 0:
                delete_list.append(trial)
        X[subject] = np.delete(X[subject], delete_list, 0)
        y[subject] = np.delete(y[subject], delete_list)
    y = [y[i] - np.min(y[i]) for i in range(len(y))]
    return X, y

##########################################################################################################

def train_test_subject(X, y, train_all=True, standardize=True):

    l = np.random.permutation(len(X[0]))
    X_test = X[0][l[:50], :, :]
    y_test = y[0][l[:50]]

    if train_all:
        X_train = np.concatenate((X[0][l[50:], :, :], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8]))
        y_train = np.concatenate((y[0][l[50:]], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]))

    else:
        X_train = X[0][l[50:], :, :]
        y_train = y[0][l[50:]]

    X_train_mean = X_train.mean(0)
    X_train_var = np.sqrt(X_train.var(0))

    if standardize:
        X_train -= X_train_mean
        X_train /= X_train_var
        X_test -= X_train_mean
        X_test /= X_train_var

    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    return X_train, X_test, y_train, y_test

##########################################################################################################


def train_test_total(X, y, standardize=True):
    X_total = np.concatenate((X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8]))
    y_total = np.concatenate((y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]))

    l = np.random.permutation(len(X_total))
    X_test = X_total[l[:50], :, :]
    y_test = y_total[l[:50]]
    X_train = X_total[l[50:], :, :]
    y_train = y_total[l[50:]]

    X_train_mean = X_train.mean(0)
    X_train_var = np.sqrt(X_train.var(0))

    if standardize:
        X_train -= X_train_mean
        X_train /= X_train_var
        X_test -= X_train_mean
        X_test /= X_train_var

    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    return X_train, X_test, y_train, y_test

##########################################################################################################

import importlib 
import preprocessing
importlib.reload(preprocessing)
from preprocessing import *
import warnings
warnings.filterwarnings('ignore')
import tensorflow.keras
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv1D,Conv2D,MaxPooling1D,Flatten,Dense,Dropout,BatchNormalization, GRU, LSTM, RNN
from tensorflow.keras import regularizers as reg
acc = []
ka = []
prec = []
recall = []

##########################################################################################################

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

##########################################################################################################

import matplotlib.pyplot as plt

def cnn_plot(conv_layers=3,conv_sizes=(64,128,256),filter_size=3, fc_layers=3,fc_sizes=(4096,2048),
        dropout=0.5,pool_size=2,init='he_uniform',act='tanh',optim='Adam',pool=True,
        reg = reg.l2(0.05),epochs=1000):
 
    classifier = Sequential()
    for i in range(conv_layers):
        classifier.add(Conv1D(conv_sizes[i], filter_size, input_shape = X_train.shape[1:],
                              activation = act,kernel_initializer=init,kernel_regularizer=reg))
       # classifier.add(BatchNormalization())
        
        if pool:
            classifier.add(MaxPooling1D(pool_size = 2))
    classifier.add(Flatten())
    for j in range(fc_layers):
        classifier.add(Dense(fc_sizes[j], activation = act,kernel_initializer=init,kernel_regularizer=reg))
        classifier.add(Dropout(dropout))       
    classifier.add(Dense(4, activation = 'softmax',kernel_initializer=init))
    classifier.compile(optimizer = optim, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    history = classifier.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=epochs,batch_size=64)
    #classifier.summary()
    y_pred = classifier.predict(X_test)
    pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
    
    classifier.summary()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

##########################################################################################################

X, y = import_data(every=False)
X_train,X_test,y_train,y_test = train_test_total(X, y)
cnn_plot(conv_layers=3,conv_sizes=(64,128,256),fc_layers=3,fc_sizes=(1024,512,256))

##########################################################################################################






