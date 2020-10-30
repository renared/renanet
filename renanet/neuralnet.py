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
import joblib

def mult_rows(A,b):
    return (A.T*b).T

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
    
    
    def grad(self, data, labels):
        N = len(data)
        L = len(self._layers)
        I = [len(self._layers[i]) for i in range(L)]
        dw = [np.zeros(self._layers[o].weights.shape) for o in range(1,L)]
        db = [np.zeros(self._layers[o].biases.shape) for o in range(1,L)]
        y = [np.zeros(len(self._layers[o])) for o in range(L)]
        for i in range(N):
            y[-1] = self(data[i])
            y_err = y[-1]-labels[i]
            magic = np.eye(I[-1])
            for l in range(L-1):
                y[-1-l-1] = self._layers[-1-l-1](data[i])
                if l>=1:
                    magic = magic @ mult_rows(self._layers[-l].weights, psi(self._layers[-l](data[i])))
                dw[-1-l] += ( (1/N)
                                *mult_rows(
                                    np.transpose(
                                        mult_rows(magic, y_err)
                                        )
                                    , psi(y[-l-1]) )
                                @ np.tile( y[-1-l-1] , (I[-1],1) )
                            )
                db[-1-l] += ( (1/N)
                               * magic.T
                               @ y_err
                               *psi(y[-1-l])
                             )
        return dw,db
    
    def grad_l(self, data, labels, l):
        #print("la couche demandée a",len(self._layers[-1-l]),"neurones")
        assert 0<=l<len(self._layers)-1
        N = len(data)
        L = len(self._layers)
        I = [len(self._layers[i]) for i in range(L)]
        dw = np.zeros(self._layers[-l-1].weights.shape)
        db = np.zeros(self._layers[-l-1].biases.shape)
        for i in range(N):
            dw += ( (1/N)
                    *mult_rows(
                        np.transpose( 
                            mult_rows(
                                multi_dot(
                                    [ mult_rows(self._layers[-1-k].weights, psi(self._layers[-1-k](data[i]))) for k in range(l)]
                                    ,I[-1] ) 
                                , self(data[i])-labels[i] )
                            )
                        , psi(self._layers[-l-1](data[i])) )
                    @ np.tile( self._layers[-1-l-1](data[i]) , (I[-1],1) )
                    )
            db += ( (1/N)
                   *np.transpose(
                       multi_dot(
                        [ mult_rows(self._layers[-1-k].weights, psi(self._layers[-1-k](data[i]))) for k in range(l)]
                        , I[-1])
                       )
                   @ (self(data[i])-labels[i])
                   *psi(self._layers[-1-l](data[i]))
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
                cost_msg = self.cost(data,labels)
                print("\rIteration {}: error {:.6f}; Time elapsed: {:.3f}s; computing gradient".format(iteration+1, cost_msg,time()-st)+32*" ", end="", flush=True)
                dw,db = self.grad(data,labels)
                
                print("\rIteration {}: error {:.6f}; Time elapsed: {:.3f}s; computing optimal step".format(iteration+1, cost_msg,time()-st)+32*" ", end="", flush=True)
                def new_cost(step):
                    shadow = copy.deepcopy(self)
                    for k in range(len(dw)):
                        shadow._layers[1+k].weights = self._layers[1+k].weights - step*dw[k]
                        shadow._layers[1+k].biases = self._layers[1+k].biases - step*db[k]
                    E = shadow.cost(data,labels)
                    del shadow
                    return E
            
                print("\rIteration {}: error {:.6f}; Time elapsed: {:.3f}s; computing optimal step".format(iteration+1, cost_msg,time()-st)+32*" ", end="", flush=True)    
                res = scipy.optimize.minimize_scalar(new_cost,method='brent',options={'maxiter':2})
                step = res.x
                if step<=0:
                    print("\rIteration {}: error {:.6f}; Time elapsed: {:.3f}s; optimal step is negative, step: {:.2e}".format(iteration+1, cost_msg,time()-st, step)+32*" ", end="", flush=True)
                    res = scipy.optimize.minimize_scalar(new_cost,bounds=(0,1),method='bounded',options={'maxiter':8})
                    step = res.x
                
                print("\rIteration {}: error {:.6f}; Time elapsed: {:.3f}s; finishing, step: {:.2e}".format(iteration+1, cost_msg,time()-st, step)+32*" ", end="", flush=True)
                for k in range(len(dw)):
                    self._layers[1+k].weights = self._layers[1+k].weights - step*dw[k]
                    self._layers[1+k].biases = self._layers[1+k].biases - step*db[k]
                layers = copy.deepcopy(self._layers)
                
        except KeyboardInterrupt:
            print("")
            self._layers = layers
    
    def learn_old(self, data, labels, iterations=1000000):
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
                    dw,db = self.grad_l(data,labels,l)
                    
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
