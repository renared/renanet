# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:07:01 2020

@author: Yann
"""

__all__ = ['Layer']

import numpy as np
from scipy.special import expit as sigmoid

class Layer:
    
    def __init__(self, parent, n):
        self._parent = parent
        self._n = n
        if parent==None:
            pass
        else :
            m = len(parent)
            self.weights = np.random.randn(n,m)   # poids des neurones de la couche : à prendre en compte par la couche suivante
            self.biases = np.random.randn(n) # biais des neurones : pour la couche suivante...
            
    def __call__(self, data):
        assert type(data) is np.ndarray
        p=len(self.getInputLayer())
        if data.shape==(p,):
            return self._eval(data)
        elif data.shape[1:]==(p,) :
            return np.array([ self._eval(d) for d in data ])
        else :
            raise TypeError("Invalid data shape: {}".format(data.shape))
    
    def _eval(self, d):
        # on ne fait pas d'assert car fonction privée
        if self._parent==None:
            return d
        else :
            return sigmoid(self.weights.dot(self._parent(d))+self.biases)
        
    def __len__(self):
        return self._n
    
    def getInputLayer(self):
        if self._parent == None:
            return self
        return self._parent.getInputLayer()

    

        