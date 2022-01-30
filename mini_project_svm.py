# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:54:03 2021

@author: Akanksha
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm


data = pd.read_csv(r"C:\Users\PC Point\Downloads\self_mob_price.csv")
print(data)

x = data.iloc[:,:-1]
y = data.iloc[:,-1]
print(x)
print(y)

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.20)

print('--------------------------')
print(X_train)
print("-------------------------")

print('--------------------------')
print(X_test)
print("-------------------------")

print('--------------------------')
print(Y_train)
print("-------------------------")


print('--------------------------')
print(Y_test)
print("-------------------------")


model = svm.SVC(kernel='linear')
model.fit(X_train,Y_train)

y_predict = model.predict(X_test)

print("\n\t Printing Individual values")
print("\n\t Accuracy : ",metrics.accuracy_score(Y_test,y_predict)*100)
print("\n\t Precision : ",metrics.precision_score(Y_test,y_predict,average = 'weighted')*100)
print("\n\t Recall : ",metrics.recall_score(Y_test, y_predict,average = 'weighted' )*100)
