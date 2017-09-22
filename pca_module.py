# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:11:29 2017

@author: Shubham Shantaram Pawar
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib .pyplot as plt

X = pd.read_csv(
    filepath_or_buffer='dataset_1.csv', 
    header=0, 
    sep=',')

X.columns=['x', 'y', 'z']
X.dropna(how="all", inplace=True) # drops the empty line at file-end

X.tail()

X_std = StandardScaler().fit_transform(X)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort()
eig_pairs.reverse()

matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), 
                      eig_pairs[1][1].reshape(3,1)))

Y = X_std.dot(matrix_w)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for item in Y:
        plt.scatter(item[0],
                    item[1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()
    plt.savefig("pca.png")
    
Output:
    
 Y
Out[14]: 
array([[-1.87969457, -0.03824594],
       [-0.9058563 ,  0.03686758],
       [ 0.20879921, -1.56793516],
       ..., 
       [ 0.4065895 ,  1.22909709],
       [-2.52246051, -1.40864796],
       [ 1.39674555, -0.48499987]])
    
