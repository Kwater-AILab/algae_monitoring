# First, let's define a RNN Cell, as a layer subclass.
import os
import time

import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
import xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', family='HCR Dotum')
# import warnings
# warnings.filterwarnings("ignore")
# tf.set_random_seed(777)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from datetime import datetime
import joblib

import itertools


class Simulation():

    
    # RandomForest Regression Algorithm
    def AL_RandomForest(trainX, trainY, testX, testY):
        rf_clf = RandomForestRegressor(n_estimators=100, random_state=15)
        rf_clf.fit(trainX, np.ravel(trainY, order="C"))
        # joblib.dump(rf_clf, )
    
        # relation_square = rf_clf.score(trainX, trainY)
        # print('RandomForest 학습 결정계수 : ', relation_square)
    
        y_pred1 = rf_clf.predict(trainX)
        y_pred2 = rf_clf.predict(testX)
    
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(range(len(testY)), testY, '-', label="Original Y")
        # ax.plot(range(len(y_pred2)), y_pred2, '-x', label="predict Y")
        # plt.legend(loc='upper right')
        # plt.show()
    
        return rf_clf, y_pred2
    
    
    # GBR(GradientBoostingRegression) Algorithm
    def AL_GradientBoosting(trainX, trainY, testX, testY):
        trainX.columns = pd.RangeIndex(trainX.shape[1])
        testX.columns = pd.RangeIndex(testX.shape[1])
    
        gbr_model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        gbr_model.fit(trainX, np.ravel(trainY, order="C"))
    
        y_pred = gbr_model.predict(trainX)
        y_pred2 = gbr_model.predict(testX)
    
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(range(len(testY)), testY, '-', label="Original Y")
        # ax.plot(range(len(y_pred2)), y_pred2, '-x', label="predict Y")
        # plt.legend(loc='upper right')
        # plt.show()
    
        return gbr_model, y_pred2
    
    
    # XGBoosting Algorithm
    def AL_XGBoosting(trainX, trainY, testX, testY):
        trainX.columns = pd.RangeIndex(trainX.shape[1])
        testX.columns = pd.RangeIndex(testX.shape[1])
    
        xgb_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=3)
        xgb_model.fit(trainX, np.ravel(trainY, order="C"))
    
        y_pred = xgb_model.predict(trainX)
        y_pred2 = xgb_model.predict(testX)
    
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(range(len(testY)), testY, '-', label="Original Y")
        # ax.plot(range(len(y_pred2)), y_pred2, '-x', label="predict Y")
        # plt.legend(loc='upper right')
        # plt.show()
    
        return y_pred2
    
    
    # LSTM Algorithm
    def AL_LSTM(trainX, trainY, testX, testY):
        # 모델 트레이닝
        model = keras.Sequential()
        # , activation='tanh'
        model.add(tf.keras.layers.Bidirectional(layers.LSTM(hiddenDim, return_sequences=True),
                                                input_shape=[seqLength, dataDim]))
        model.add(tf.keras.layers.Dropout(drop_rate))
        model.add(tf.keras.layers.Bidirectional(layers.LSTM(10)))
        model.add(layers.Dense(1))
    
        # optimizer 설정
        opt = optimizers.Adam(lr=lr)
    
        # 모델 학습과정 설정
        model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    
        # 모델 학습습
        his = model.fit(trainX, trainY, epochs=100, batch_size=16, verbose=0,
                        validation_split=0.2,
                        callbacks=cbs)
    
        # 모델 테스트
        y_pred = model.predict(trainX)
        y_pred2 = model.predict(testX)
    
        return model, y_pred2