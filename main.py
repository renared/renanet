# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:46:45 2020

@author: Yann
"""

import renanet
import numpy as np
import graph

rena = renanet.NeuralNet(2,10,1)

X = np.array([[0,0.2],[1.72,0.32],[0.98,1.26],[-2,1],[-0.68,2.58],
              [-1.76,-0.74],[1.02,-1.52],[-0.34,-2.76],[0,-1],[-3.06,-0.32]])
C = np.array([[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]])


#perceptron.learn(X,C,iterations=16)

#X_test, C_test
# q=0
# for i in range(len(X_test)):
#     if perceptron(X_test[i]) == C_test[i]:
#         q+=1
# print(q/len(X_test), "% success rate")