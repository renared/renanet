# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 03:46:34 2020

@author: Yann
"""

import renanet
import numpy as np
import graph
import utility

rena = renanet.NeuralNet(400,40,10)

X,C = utility.chargerFichiersManuscrits(r'/home/yann/Desktop/bourdinyrichrobi/data_chamilo/Data/DigitTrain_%d.mat')
X_test,C_test = utility.chargerFichiersManuscrits(r'/home/yann/Desktop/bourdinyrichrobi/data_chamilo/Data/DigitTest_%d.mat')

#rena.load("ch.npy")
rena.learn(X,C)

utility.test(rena,X_test,C_test,mode='max')
