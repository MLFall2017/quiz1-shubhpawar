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

variance_vec = np.var(X, axis=0)
covariance_xy = np.cov(X.iloc[:,0], X.iloc[:,1])
covariance_yz = np.cov(X.iloc[:,1], X.iloc[:,2])                

X_std = StandardScaler().fit_transform(X)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort()
eig_pairs.reverse()

matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), 
                      eig_pairs[1][1].reshape(3,1)))

Y = X_std.dot(matrix_w)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(Y[:,0],Y[:,1])
fig.show()
plt.savefig("pca.png")
    
