# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 11:25:46 2020

@author: LAVEENA
"""

import scipy.io
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense

#Loading the data
data = scipy.io.loadmat('C://Users/LAVEENA/Desktop/Multi_class_classification.mat')
X = data['X']
y = data['y']

#converting labels to one hot encodings
y = to_categorical(y)
#splitting data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#defining structure of neural net
model = Sequential()
model.add(Dense(200, input_dim=400, activation='relu')) #input shape is mx400, and outputs matrix of size mx200
model.add(Dense(12, activation='relu')) #output shape is  mx12
model.add(Dense(11, activation='softmax')) #outputs  mx11, where m is no. of samples
model.summary()

#fitting the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x = X_train, y=y_train, batch_size=100, epochs=20, verbose=2, validation_split = 0.0, shuffle = True)

_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(test_acc)