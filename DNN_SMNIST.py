# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:50:18 2020

@author: ManavChordia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

df_train = pd.read_csv("sign_mnist_train.csv")
x_train = df_train.iloc[:, 1:785]
y_train = df_train.iloc[:, 0:1]
df_test = pd.read_csv("sign_mnist_test.csv")
x_test = df_test.iloc[:, 1:785]
y_test = df_test.iloc[:, 0:1]

one_hot_labels = np.zeros((27455,26))
for i in range(27455):
    one_hot_labels[i, y_train.iloc[i]] = 1
    
one_hot_labels_test = np.zeros((7172,26))
for i in range(7172):
    one_hot_labels_test[i, y_train.iloc[i]] = 1
    
lr = 0.0005
epochs = 100
batch_size = 323

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

#Make placeholders for training data
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 26])

#declare weights
W1 = tf.Variable(tf.random_normal([784, 30], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([30]), name='b1')
W2 = tf.Variable(tf.random_normal([30, 30], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([30]), name='b2')
W3 = tf.Variable(tf.random_normal([30, 26], stddev=0.03), name='W3')
b3 = tf.Variable(tf.random_normal([26]), name='b3')

#forward propagation
Z1 = tf.add(tf.matmul(x, W1), b1)
A1 = tf.nn.relu(Z1)                                    
Z2 = tf.add(tf.matmul(A1, W2), b2)
A2 = tf.nn.relu(Z2)                                    
Z3 = tf.add(tf.matmul(A2, W3), b3)  
A3 = tf.nn.softmax(Z3)

#calculating cost
A3_ = tf.clip_by_value(A3, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(A3_)
                + (1 - y) * tf.log(1 - A3_), axis=1))


optimiser = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy)

saver = tf.train.Saver()

init_op = tf.global_variables_initializer()

#accuracy assesing
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(A3, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.device("/device:GPU:1"):
# start the session
    with tf.Session() as sess:
        
        # initialise the variables
        sess.run(init_op)
        w1 = sess.run(W1)
        #total_batch = 85
        for epoch in range(20000):
            avg_cost = 0
            _, c, preds = sess.run([optimiser, cross_entropy, A3], 
                                       feed_dict={x: x_train, y: one_hot_labels})
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(c))
        w1 = sess.run(W1)
        w2 = sess.run(W2)
        w3 = sess.run(W3)
        b1 = sess.run(b1)
        b2 = sess.run(b2)
        b3 = sess.run(b3)
        saver.save(sess, "smnist_trained_model")



    
 
    
    