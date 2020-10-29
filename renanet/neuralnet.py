# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:52:25 2020

@author: Yann
"""

__all__ = ['NeuralNet']

import numpy as np

from . import layer
from scipy.special import expit as sigmoid
import uuid
import scipy.optimize
import copy
from time import time

def multi_dot(arr,default):
    if len(arr)==0:
        return np.eye(default)
    if len(arr)==1:
        return arr[0]
    return np.linalg.multi_dot(arr)

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
    
    def __len__(self):
        return len(self._layers[-1])
    
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
        assert 0<=l<len(self._layers)-1
        N = len(data)
        L = len(self._layers)
        I = [len(self._layers[i]) for i in range(L)]
        dw = np.zeros(self._layers[-l-1].weights.shape)
        db = np.zeros(self._layers[-l-1].biases.shape)
        for i in range(N):
            # a = (1/N)*np.diag(psi(self._layers[-l](data[i]))) 
            # b = np.ones( (I[-1-l],I[-1]) )
            # c = b @ np.diag(self(data[i])-labels[i])
            # d = multi_dot([np.diag(psi(self._layers[-1-k](data[i]))) @ self._layers[-1-k].weights for k in range(l-1)]
            #               ,I[-1])
            # e = np.transpose(c @ d)
            # print(a.shape, e.shape)
            # f = a @ e
            # g = f @ np.diag(self._layers[-1-l](data[i]))
            # dw += g
            
            dw += ( (1/N)
                    *np.diag(psi(self._layers[-l-1](data[i])))
                    @ np.transpose( 
                        np.ones((I[-1-l-1],I[-1]))
                        @ np.diag(self(data[i])-labels[i])
                        @ multi_dot(
                            [np.diag(psi(self._layers[-1-k](data[i]))) @ self._layers[-1-k].weights for k in range(l)]
                            ,I[-1] ) 
                        )
                    @ np.diag( self._layers[-1-l-1](data[i]) )
                    )
            db += ( (1/N)
                   *(psi(self._layers[-1-l](data[i])))
                   @ np.transpose(
                       np.diag(self(data[i])-labels[i])
                       @ multi_dot(
                        [np.diag(psi(self._layers[-1-k](data[i]))) @ self._layers[-1-k].weights for k in range(l)]
                        , I[-1])
                       )
                   )
        return dw,db
    
    def learn(self, data, labels, iterations=1000000):
        assert type(data) is np.ndarray
        assert data.shape[1:] == (len(self._layers[0]),)
        print("Started learning")
        print("You can Keyboard Interrupt at any moment without messing anything up.")
        layers = copy.deepcopy(self._layers)
        st = time()
        try:
            for iteration in range(iterations):
                for l in range(len(self._layers)-1):
                    # {:.1f}
                    print("\rIteration {}: error {:.6f}; Time elapsed: {:.3f}s".format(iteration+1, self.cost(data,labels),time()-st), end="", flush=True)
                    dw,db = self.grad(data,labels,l)
                    
                    def new_cost(step):
                        shadow = copy.deepcopy(self._layers)
                        self._layers[-1-l].weights = self._layers[-1-l].weights - step*dw
                        self._layers[-1-l].biases = self._layers[-1-l].biases - step*db
                        E = self.cost(data,labels)
                        self._layers = shadow
                        return E
                
                    res = scipy.optimize.minimize_scalar(new_cost)
                    step = res.x
                    self._layers[-1-l].weights = self._layers[-1-l].weights - step*dw
                    self._layers[-1-l].biases = self._layers[-1-l].biases - step*db
                layers = copy.deepcopy(self._layers)
        except KeyboardInterrupt:
            self._layers = layers
                
        
    def save(self,filename):
        arr = []
        for i in range(1,len(self._layers)):
            arr.append(self._layers[i].weights)
            arr.append(self._layers[i].biases)
        np.save(filename,arr)
    
    def load(self,filename):
        arr = np.load(filename,allow_pickle=True)
        if len(arr)!=2*(len(self._layers)-1):
            raise Exception("Saved renanet doesn't match dimensions.")
        for i in range(1,len(self._layers)):
            if (self._layers[i].weights.shape != arr[2*(i-1)].shape
                or self._layers[i].biases.shape != arr[2*(i-1)+1].shape ):
                raise Exception("Can't load saved net: the saved layers don't have the same shapes as the net's layers.")
        for i in range(len(self._layers)):
            self._layers[i].weights = arr[2*(i-1)]
            self._layers[i].biases = arr[2*(i-1)+1]
