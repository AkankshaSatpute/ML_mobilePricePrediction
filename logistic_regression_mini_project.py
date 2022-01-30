# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:28:38 2021

@author: Akanksha
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
sns.set()

data = pd.read_csv(r"C:\Users\PC Point\Downloads\self_mob_price.csv")
print(data)

x = data.iloc[:, :-1] 
y = data.iloc[:, -1] 
x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


lreg = LogisticRegression()
lreg.fit(x_train, y_train)
y_pred = lreg.predict(x_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy of the Logistic Regression Model:= ",accuracy)
print("\n\t Precision : ",metrics.precision_score(y_test,y_pred,average = 'weighted')*100)
print("\n\t Recall : ",metrics.recall_score(y_test, y_pred,average = 'weighted' )*100)

(unique, counts) = np.unique(y_pred, return_counts=True)
price_range = np.asarray((unique, counts)).T
print(price_range)