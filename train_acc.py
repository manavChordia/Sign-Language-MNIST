# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 08:49:10 2020

@author: ManavChordia
"""
import pandas as pd
import numpy as np

#trainning accuracy

df_train = pd.read_csv("sign_mnist_train.csv")
y_train = df_train.iloc[:, 0:1]
one_hot_labels = np.zeros((27455,26))
for i in range(27455):
    one_hot_labels[i, y_train.iloc[i]] = 1
    
predictions_train = np.load('training_predictions.npy')  

preds_train = np.around(predictions_train)

acc = one_hot_labels - preds_train
print(acc.sum())

num1=0
for i in acc:
    for j in i:
        if j==1:
            num1=num1+1
            
num_1=0
for i in acc:
    for j in i:
        if j==1:
            num_1=num_1+1

#Training accuracy = 99.9927%!!!