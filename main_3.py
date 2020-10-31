# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 03:46:34 2020

@author: Yann
"""

import renanet
import numpy as np
import graph
import utility
import csv

rena = renanet.NeuralNet(9,81,81,1)

def readData(csv_file):
    rows = []
    with open(csv_file, 'r') as file:
        csv_dict = csv.DictReader(file)
        for row in csv_dict:
            rows.append(dict(row))

    X = []
    C = []
    for row in rows:
        if 'N/A' not in row.values() :
            if row['WRank']=='' or row['LRank']=='' or row['WPts']=='' or row['LPts']=='' or row['AvgW']=='' or row['AvgL']=='':
                continue
            x = []
            c = []
            x.append(0 if row['Court']=='Indoor' else 1)
            x.append(0 if row['Surface']=='Hard' else (0.5 if row['Surface']=='Clay' else 1))
            x.append(0 if row['Best of']=='3' else 1)
            if np.random.rand()<0.5:
                x.append(int(row['WRank']))
                x.append(int(row['LRank']))
                x.append(int(row['WPts']))
                x.append(int(row['LPts']))
                x.append(float(row['AvgW']))
                x.append(float(row['AvgL']))
                c.append(0)
            else :
                x.append(int(row['LRank']))
                x.append(int(row['WRank']))
                x.append(int(row['LPts']))
                x.append(int(row['WPts']))
                x.append(float(row['AvgL']))
                x.append(float(row['AvgW']))
                c.append(1)
            X.append(x)
            C.append(c)
    X = np.array(X)
    C = np.array(C)
    return X,C

def rente(net,X,C):
    mise_par_pari = 1.0
    compte = 0.0
    for i in range(len(X)):
        compte-=mise_par_pari
        y = 0 if net(X[i])<0.5 else 1
        if (C[i]==y):
            cote = X[i][-2] if y==0 else X[i][-1]
            compte += cote*mise_par_pari
    return compte

#rena.load("ch.npy")
X,C = readData('2017.csv')
_X,_C = readData('2018.csv')
X,C = np.concatenate([X,_X]),np.concatenate([C,_C])
_X,_C = readData('2019.csv')
X,C = np.concatenate([X,_X]),np.concatenate([C,_C])
rena.learn(X,C)
X_test,C_test = readData('2020.csv')
utility.test(rena,X_test,C_test)
print("En misant 1€ par pari, on gagne {:.2f}€".format(rente(rena,X_test,C_test)))