# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:40:19 2020

@author: ManavChordia
"""

import pandas as pd
import numpy as np

df_test = pd.read_csv("sign_mnist_test.csv")
x_test = df_test.iloc[:, 1:785]
y_test = df_test.iloc[:, 0:1]

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def relu(X):
   return np.maximum(0,X)

one_hot_labels_test = np.zeros((7172,26))
for i in range(7172):
    one_hot_labels_test[i, y_test.iloc[i]] = 1

W1 = np.load('W1.npy')
W2 = np.load('W2.npy')
W3 = np.load('W3.npy')
b1 = np.load('b1.npy')
b2 = np.load('b2.npy')
b3 = np.load('b3.npy')


z1 = np.dot(x_test, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
a2 = relu(z2)
z3 = np.dot(a2, W3) + b3
a3 = softmax(z3)

preds_test = np.around(a3)
opt = (one_hot_labels_test-preds_test)
print(np.sum(opt))

num1=0
for i in opt:
    for j in i:
        if j==1:
            num1=num1+1
            
num_1=0
for i in opt:
    for j in i:
        if j==1:
            num_1=num_1+1

