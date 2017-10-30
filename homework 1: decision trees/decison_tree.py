# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:52:02 2017

@author: opeide
"""
import numpy as np
from collections import Counter



class DecisionTree():
    def __init__(self, data, depth_lim):
        #build decision tree
        self._build_tree(data, 0, depth_lim)
              
              
    def _build_tree(self, group, current_depth, depth_lim):
        if current_depth >= depth_lim:
            return        
        split_result = self._split_once(group)
        if split_result['end']:
            return
        print('left at depth {} if x{} < {}'.format(current_depth, split_result['dim'], split_result['right_lim']))
        self._build_tree(split_result['left'], current_depth+1, depth_lim)
        self._build_tree(split_result['right'], current_depth+1, depth_lim)
    
    
    def _split_once(self, data):
        print('NEW SPLIT')
        #find best index to split a given data set  
        best_gini_ch = 0
        best_dim = 0   
        best_index = 0  
        data_best_dim_sorted = []
                  
        for dim in range(len(data[0])-1):
            #sort by position in chosen dimension
            data_dim_sorted = sorted(data, key= lambda x: x[dim])
            y_dim_sorted = [row[3] for row in data_dim_sorted]  
            #test the change in label-purity for every possible way to split the data set in two                        
            for i in range(len(y_dim_sorted)):
                y_left = y_dim_sorted[0:i]
                y_right = y_dim_sorted[i:-1]
                gini_ch = self._weighted_gini_change(y_dim_sorted, y_left, y_right)
                if gini_ch > best_gini_ch:
                    best_gini_ch = gini_ch
                    best_index = i
                    best_dim = dim
                    data_best_dim_sorted = data_dim_sorted  
        if best_gini_ch == 0:
            print('cant improve')
            return {'end':True}
        print('SPLIT INTO: ',[row[3] for row in data_best_dim_sorted[0:best_index]], [row[3] for row in data_best_dim_sorted[best_index:-1]])
        print('gini:',self._calc_gini([row[3] for row in data_best_dim_sorted[0:best_index]]), self._calc_gini([row[3] for row in data_best_dim_sorted[best_index:-1]]))
        return {'end': False, 'dim':best_dim, 'right_lim':data_best_dim_sorted[best_index][best_dim],
        'left':data_best_dim_sorted[0:best_index], 'right':data_best_dim_sorted[best_index:-1]}
                
        
    def _calc_gini(self, y_group):
        success_prob = 0.0        
        counter = Counter(y_group)
        tot_y = float(len(y_group))        
        for y in counter:
            success_prob += (float(counter[y])/tot_y)**2
        return 1.0-success_prob
    
    def _weighted_gini_change(self, start, left, right):
        w_left = float(len(left)) / float(len(start))
        w_right = float(len(right)) / float(len(start))
        gini_start = self._calc_gini(start)
        gini_left = self._calc_gini(left)
        gini_right = self._calc_gini(right)
        
        gini_ch = gini_start -  w_left*gini_left - w_right*gini_right  
        #print('{}: {} -> {} | {}'.format(gini_ch,start,left,right))
        return gini_ch
            
if __name__ == '__main__':
    all_data = np.loadtxt('01_homework_dataset.csv', delimiter=',', skiprows=1)

    tree = DecisionTree(all_data, depth_lim=2)
    