#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:11:16 2020

@author: rey10
"""
#import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils import to_categorical
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

#load dataset

dataframe = pd.read_csv("final2.csv", header='infer')
dataframe2 = pd.read_csv("Withi_river.csv", header='infer')
data1 = pd.read_excel("thiyanaru-attachments/Deduru Oya at Chilaw_WL.xlsx", header='infer') 

dataset = dataframe.values
np.random.shuffle(dataset)

X1 = dataset[1:3005,1]
X1 = X1.reshape((3004,1))
X2 = dataset[1:3005,5]
X2 = np.fromstring( X2, dtype=np.float)
X2 = X2.reshape((3004,1))

X =np.concatenate((X1,X2),axis=1)
y = dataframe2.values


# Create train/test
x_train, x_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

# Build the neural network
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu')) # Hidden 1
model.add(Dense(64, activation='relu')) # Hidden 2
model.add(Dense(32, activation='relu')) # Hidden 3
model.add(Dense(32, activation='relu')) # Hidden 4
model.add(Dense(1)) # Output
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=[metrics.mae, metrics.categorical_accuracy])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto', restore_best_weights=True)
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=2,epochs=1000)


from sklearn import metrics

# Predict
pred = model.predict(x_test)

# Measure MSE error.  
score = metrics.mean_squared_error(pred,y_test)
print("Final score (MSE): {}".format(score))


import numpy as np

# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Final score (RMSE): {}".format(score))


def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
 
# Plot the chart
chart_regression(pred.flatten(),y_test)    

# =============================================================================
# 
# # Plotting Results
# import matplotlib.pyplot as plt
# 
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# 
# plt.plot(epochs, acc, 'b', label='Training acc')
# plt.plot(epochs, val_acc, 'g', label='Validation acc')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# 
# plt.title('Training and validation accuracy')
# plt.legend()
# fig = plt.figure()
# fig.savefig('acc.png')
# 
# 
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'g', label='Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and validation loss')
# 
# plt.legend()
# plt.show()
# =============================================================================
