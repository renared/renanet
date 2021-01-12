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
import pickle
import collections
import matplotlib.pyplot as plt

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
    
    def validate(self,X,C,method='argmax'):
        '''returns success rate of the perceptron to guess the classes C of data X'''
        if (len(X)==0): return float("+inf")
        Y = self(X)
        M = np.argmax(Y,axis=1)
        N = np.argmax(C,axis=1)
        n_good_guesses = np.sum(1*(M==N))
        return n_good_guesses/len(X)
    
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
    
    def learn(self, data, labels, iterations=1000000, validation_set=([],[])):
        assert type(data) is np.ndarray
        assert data.shape[1:] == (len(self._layers[0]),)
        print("Started learning")
        print("You can Keyboard Interrupt at any moment without messing anything up.")
        layers = copy.deepcopy(self._layers)
        st = time()
        best_E = float("+inf")
        err_hist = []
        best_success = 0
        try:
            w,b=self.get_values()
            layers = []
            for iteration in range(iterations):
                
                # plt.scatter([iteration],[self.validate(data,labels)],c='green')
                # plt.scatter([iteration],[self.validate(*validation_set)],c='blue')
                
                # error
                err = self.cost(data,labels)
                err_hist.append(err)
                
                # success rates
                success_train = self.validate(data,labels)
                success_validate = self.validate(*validation_set)
                
                if err<=best_E:
                    best_E=err
                    if success_validate==float("+inf"): # revient à dire qu'il n'y a pas de validation set
                        layers = copy.deepcopy(self._layers)
                    
                if success_validate > best_success:
                    best_success = success_validate
                    layers = copy.deepcopy(self._layers)
                    
                def __msg(text):
                    print("\rIteration {}: error {:.6f}, train {:.2f}%, validate {:.2f}%; Time elapsed: {:.3f}s; {}".format(iteration+1, err, success_train*100, success_validate*100, time()-st, text)+32*" ", end="", flush=True)
                
                __msg("computing gradient")
                dw,db = self.grad(data,labels)
                
                def new_cost(step):
                    shadow = copy.deepcopy(self)
                    shadow.step_grad_descent(dw, db, step)
                    E = shadow.cost(data,labels)
                    del shadow
                    return E
            
                __msg("computing optimal gradient step")
                res = scipy.optimize.minimize_scalar(new_cost,method='brent',options={'maxiter':16})
                step = res.x
                if step<=0:
                    __msg("optimal gradient step is negative")
                    res = scipy.optimize.minimize_scalar(new_cost,bounds=(0,1),method='bounded',options={'maxiter':16})
                    step = res.x
                    
                self.step_grad_descent(dw, db, step)
                    
                # __msg("computing optimal noise step")
                dw,db = [np.random.randn(*x.shape) for x in dw], [np.random.randn(*x.shape) for x in db]
                # res = scipy.optimize.minimize_scalar(new_cost,method='brent',options={'maxiter':16})
                # step = res.x
                step=0.015
                
                self.step_grad_descent(dw, db, step)
                
                __msg("finishing iteration")
                
        except KeyboardInterrupt:
            print("\nError: {:.6f}".format(best_E))
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
                
    def set_values(self,w,b):
        assert len(w)==len(b)==len(self._layers)-1
        for k in range(len(w)):
            self._layers[1+k].weights = w[k]
            self._layers[1+k].biases = b[k]
            
    def get_values(self):
        w=[]
        b=[]
        for k in range(len(self._layers)-1):
            w.append(self._layers[1+k].weights)
            b.append(self._layers[1+k].biases)
        return w,b
    
    def step_grad_descent(self,dw,db,step):
        assert len(dw)==len(db)==len(self._layers)-1
        for k in range(len(dw)):
            self._layers[1+k].weights = self._layers[1+k].weights - step*dw[k]
            self._layers[1+k].biases = self._layers[1+k].biases - step*db[k]
        
    def save(self,filename):
        arr = []
        for i in range(1,len(self._layers)):
            arr.append(self._layers[i].weights)
            arr.append(self._layers[i].biases)
        pickle.dump( arr, open( filename, "wb" ) )
        # np.save(filename,arr,allow_pickle=True)
    
    def load(self,filename):
        arr = pickle.load( open( filename, "rb" ) )
        # arr = np.load(filename,allow_pickle=True)
        if len(arr)!=2*(len(self._layers)-1):
            raise Exception("Saved renanet doesn't match dimensions.")
        for i in range(1,len(self._layers)):
            if (self._layers[i].weights.shape != arr[2*(i-1)].shape
                or self._layers[i].biases.shape != arr[2*(i-1)+1].shape ):
                raise Exception("Can't load saved net: the saved layers don't have the same shapes as the net's layers.")
        for i in range(len(self._layers)):
            self._layers[i].weights = arr[2*(i-1)]
            self._layers[i].biases = arr[2*(i-1)+1]
