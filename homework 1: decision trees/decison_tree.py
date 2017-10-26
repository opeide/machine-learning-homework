# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:52:02 2017

@author: opeide
"""
import numpy as np
from collections import Counter




class decision_tree():
    def __init__(self, data, depth_lim):

    #find best index to split a given data set
        for dim in range(len(data[0])-1):
            data_dim_sorted = sorted(data, key= lambda x: x[dim])
            y_dim_sorted = [row[3] for row in data_dim_sorted]            
            avg_gini_for_split = []            
            for i in range(len(y_dim_sorted)):
                avg_gini_for_split.append(0.5*(self._calc_gini(y_dim_sorted[0:i]) + self._calc_gini(y_dim_sorted[i:-1])))
        best_split_index = avg_gini_for_split.index(min(avg_gini_for_split))
        print(best_split_index)
       
    def _calc_gini(self, y_group):
        success_prob = 0.0        
        counter = Counter(y_group)
        tot_y = float(len(y_group))        
        for y in counter:
            success_prob += (float(counter[y])/tot_y)**2
        return 1.0-success_prob
        
            
if __name__ == '__main__':
    all_data = np.loadtxt('01_homework_dataset.csv', delimiter=',', skiprows=1)

    
    tree = decision_tree(all_data, depth_lim=2)
    