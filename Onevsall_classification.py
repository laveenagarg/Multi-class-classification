# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:01:30 2020

@author: LAVEENA
"""

import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#Loading the data
data = scipy.io.loadmat('C://Users/LAVEENA/Desktop/Multi_class_classification.mat')
X = data['X']
y = data['y']

#Prediction for class with label 1
y1 = (y==1)  #defining new labels
y1 = y1.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2)

model = LogisticRegression().fit(X_train,y_train)
y_pred = model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix) #prints confusion metrix
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


#Prediction for class with label 2
y2 = (y==2)  #defining new labels
y2 = y2.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2)

model = LogisticRegression().fit(X_train,y_train)
y_pred = model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix) #prints confusion metrix
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


#and so on we can do this for each class