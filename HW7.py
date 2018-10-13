#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:16:22 2018

@author: yizhouwang
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Part 1
data=pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/wine/wine.data',
         header=None)
data.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
X=data.iloc[:,:-1].values
y=data.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.1, stratify=y,random_state=1)

params=[5,10,50,100,500]

for i in params:
    print('N_Estimators: ', i )
    RF=RandomForestClassifier(criterion='gini', max_depth=4, n_estimators=i,random_state=1)
    RF.fit(X_train,y_train)
    scores = cross_val_score(estimator=RF, X=X_train, y=y_train, cv=10, n_jobs=1)
    print('In_Sample accuray: ', np.mean(scores))
    y_pred=RF.predict(X_test)
    outofsample =accuracy_score(y_test,y_pred)
    print('Out_Sample accuray: ', outofsample)    


# Part 2
feat_labels = data.columns[1:]
rf=RandomForestClassifier(n_estimators=100,
                          random_state=1)
rf.fit(X_train,y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print("My name is Yizhou Wang")
print("My NetID is: yizhouw4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")