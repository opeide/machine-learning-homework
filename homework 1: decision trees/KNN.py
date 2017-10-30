# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 01:18:42 2017

@author: opeide
"""
import numpy as np

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1-x2)

def get_neighbors_labels(X_train, y_train, x_new, k):
    distances = np.array([euclidean_distance(x_new, x) for x in X_train])
    nearest_indexes = distances.argsort()
    y_nearest = [y_train[i] for i in nearest_indexes[0:k]]
    return y_nearest

def get_majority_label(neighbors_labels):
    freq = {}
    for label in neighbors_labels:
        freq[label] = freq.get(label, 0) + 1
    max_f = max(freq.values())
    max_f_label = filter(lambda key: freq[key]==max_f, freq.keys())
    return min(max_f_label)

def get_average_label(neighbors_labels):
    return float(sum(neighbors_labels))/float(len(neighbors_labels))
    
def predict(X_train, y_train, X_test, k):
    for x_new in X_test:
        neighbors = get_neighbors_labels(X_train, y_train, x_new, k)
        pred = get_majority_label(neighbors)
        print('{} majority says {}'.format(x_new, pred))
        pred = get_average_label(neighbors)
        print('{} average says {}'.format(x_new, pred))


all_data = np.loadtxt('01_homework_dataset.csv', delimiter=',', skiprows=1)
x = [np.array(row[0:3]) for row in all_data]
y = [row[3] for row in all_data]
xa = np.array([4.1, -0.1, 2.2])
xb = np.array([6.1, 0.4, 1.3])
predict(x, y, [xa,xb], k=3)