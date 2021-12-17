# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:31:55 2018

@author: epinsky
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  

# https://stackoverflow.com/questions/34408027/how-to-allow-sklearn-k-nearest-neighbors-to-take-custom-distance-metric

data = pd.DataFrame(
        {'id': [ 1,2,3,4,5,6],
         'Label': ['green', 'red', 'red', 'green', 'green', 'red'],
         'X': [1, 6, 7, 10, 10, 15], 
         'Y': [2, 4, 5, -1, 2, 2 ]},
         columns = ['id', 'Label', 'X', 'Y']
        )

X = data[['X','Y']].values
Y = data[['Label']].values

# default kNN with Euclidean distance (add p = 1 to get Street)
knn_classifier_1 = KNeighborsClassifier(n_neighbors=3) 
knn_classifier_1.fit(X,Y)

new_instance = np.asmatrix([3, 2])
prediction = knn_classifier_1.predict(new_instance)
print('new_label for classifier 1: ', prediction[0])


# knn with Minkowski metric, any p not necessarily an integer
def minkowski_p(a,b,p):
    return np.linalg.norm(a-b, ord=p)

p = 1.5
knn_Minkowski_p = KNeighborsClassifier(n_neighbors=3, 
                                        metric = lambda a,b: minkowski_p(a,b,p) ) 
knn_Minkowski_p.fit(X,Y)
new_instance = np.asmatrix([3, 2])
prediction = knn_Minkowski_p.predict(new_instance)
print('new_label for Minkowski_p: ', prediction[0])








