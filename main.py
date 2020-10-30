# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:46:45 2020

@author: Yann
"""

import renanet
import numpy as np
import graph
import utility

rena = renanet.NeuralNet(2,10000,1)

# X = np.array([[0,0.2],[1.72,0.32],[0.98,1.26],[-2,1],[-0.68,2.58],
#               [-1.76,-0.74],[1.02,-1.52],[-0.34,-2.76],[0,-1],[-3.06,-0.32]])
# C = np.array([[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]])

X = utility.lectureMatData(r'D:\Yann\Desktop\bourdinyrichrobi\data_chamilo\DataSimulation\DataTrain_2Classes_Perceptron_2.mat', nomColonne='data')
C = utility.lectureMatData(r'D:\Yann\Desktop\bourdinyrichrobi\data_chamilo\DataSimulation\DataTrain_2Classes_Perceptron_2.mat', nomColonne='c')

X_test = utility.lectureMatData(r'D:\Yann\Desktop\bourdinyrichrobi\data_chamilo\DataSimulation\DataTest_2Classes_Perceptron_2.mat', nomColonne='dataTest')
C_test = utility.lectureMatData(r'D:\Yann\Desktop\bourdinyrichrobi\data_chamilo\DataSimulation\DataTest_2Classes_Perceptron_2.mat', nomColonne='cTest')

rena.learn(X,C)
utility.show(rena,X_test,C_test,utility.test(rena,X_test,C_test))

# X,C = utility.chargerFichiersManuscrits(r'D:\Yann\Desktop\bourdinyrichrobi\data_chamilo\Data\DigitTrain_%d.mat')