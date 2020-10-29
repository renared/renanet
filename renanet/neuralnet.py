# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:52:25 2020

@author: Yann
"""

__all__ = ['NeuralNet']

import numpy as np

from . import layer
from scipy.special import expit as sigmoid
from numpy.linalg import multi_dot

def psi(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNet:
    
    def __init__(self, *args):
        for x in args:
            assert type(x) is int
            assert x>0
        
        self._layers = []
        for n in args:
            self._layers.append(layer.Layer(self._layers[-1] if len(self._layers)>0 else None, n))
        
    def __call__(self, data):
        # assert type(data) is np.ndarray
        # assert data.shape[1:] == (len(self._layers[0]),)
        return self._layers[-1](data)
    
    def cost(self,data,labels):
        assert type(data)==type(labels)==np.ndarray
        assert len(data)==len(labels)
        assert data.shape[1:] == (len(self._layers[0]),)
        assert labels.shape[1:] == (len(self._layers[-1]),)
        return np.sum( (self(data) - labels)**2 )/len(data)/2
    
    # def grad(self,data,labels):
    #     psi = lambda x : sigmoid(x)*(1-sigmoid(x))
    #     ylLx = self(data)
    #     ylL1x = self._layers[-2](data)
    #     #return (1/len(data))*self._layers[-1].grad(data,labels)
    #     dw=0
    #     for i in range(len(data)):
    #         dw += np.diag(ylLx[i]-labels[i]).dot(np.diag(psi(ylLx[i]))).dot(np.ones((len(ylLx[i]),len(ylL1x[i])))).dot(np.diag(ylL1x[i]))
    #     dw /= len(data)
    #     db=0
    #     for i in range(len(data)):
    #         db += (ylLx[i]-labels[i])*psi(ylLx[i])
    #     db /= len(data)
    #     return dw,db
    
    def grad(self, data, labels, l):
        #print("la couche demandée a",len(self._layers[-1-l]),"neurones")
        assert 1<=l<len(self._layers)
        N = len(data)
        L = len(self._layers)
        I = [len(self._layers[i]) for i in range(L)]
        dw = np.zeros(self._layers[-l].weights.shape)
        db = np.zeros(self._layers[-l].biases.shape)
        for i in range(N):
            dw += ( (1/N)
                   *np.diag(psi(self(data[i])))
                   @ np.transpose( 
                       np.ones((I[-1-l],I[-1]))
                       @ np.diag(self(data[i])-labels[i])
                       @ ( multi_dot([
                           *[np.diag(psi(self._layers[-1-k](data[i]))) @ self._layers[-1-k-1].weights for k in range(l-1)]
                           ]) if l>1 else np.eye(I[-1]) )
                       )
                   @ np.diag( self._layers[-1-l](data[i]) )
                   )
        return dw,db
    
    def learn(self, data, labels, iterations=16):
        assert type(data) is np.ndarray
        assert data.shape[1:] == (len(self._layers[0]),)
        
        for iteration in range(iterations):
            for l in range(len(self._layers)-1):
                #print(l)
                self._layers[-1-l].weights = self._layers[-1-l].weights - 0.1*self.grad(data,labels,l+1)[0]
        
    def save(self):
        pass
    
    def load(self):
        pass
