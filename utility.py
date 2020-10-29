# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:08:25 2020

@author: Yann

Utility
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

import graph

def lectureMat2D(fichierMat, nomColonneX='data', nomColonneC='c', output_neurons=1):
    '''
    Read a .mat file with 2D data

    Parameters
    ----------
    fichierMat : str
        filename.
    nomColonneX : str, optional
        Name of the column corresponding to the data. The default is 'data'.
    nomColonneC : str, optional
        Name of the column corresponding to the expected output of neural net. The default is 'c'.
    output_neurons : int, optional
        Number of neurons in the output layer. The default is 1.

    Returns
    -------
    X : tensor
        data, shape (n,2,1).
    C : TYPE
        expected output of the neural net, shape (n, output_neurons).
    '''
    matr = loadmat(fichierMat)
    X = matr[nomColonneX].transpose()
    X = X.reshape(len(X), 2, 1)
    C = matr[nomColonneC][0]
    C = C.reshape(len(C), output_neurons)
    return X, C

def lectureMatData(fichierMat, nomColonne='imgs', shape=None):
    '''
    Generic function to read a .mat file with data in a column.

    Parameters
    ----------
    fichierMat : str
        filename.
    nomColonne : str, optional
        Name of the column to read. The default is 'imgs'.
    shape : tuple, optional
        Shape to reshape the data to. The default is None.

    Returns
    -------
    X
        data tensor.
    '''
    matr = loadmat(fichierMat)
    X = matr[nomColonne].transpose()
    return X.reshape(shape)

def lectureMatManuscrit2Classes(fichierClasse0, fichierClasse1, limit=0):
    '''
    Read the .mat data files corresponding to handwritten digits

    Parameters
    ----------
    fichierClasse0 : str
        filename.
    fichierClasse1 : str
        filename.
    limit : int, optional
        Restrict each file to a certain number of entries. The default is 0 (no limit).

    Returns
    -------
    X : 
        data tensor.
    C : 
        expected output of the output layer of the neural net.
    '''
    X0 = lectureMatData(fichierClasse0, nomColonne='imgs')
    if limit > 0:
        X0 = X0[:limit]
    C0 = np.zeros(len(X0)).reshape(len(X0), 1)
    X1 = lectureMatData(fichierClasse1, nomColonne='imgs')
    if limit > 0:
        X1 = X1[:limit]
    C1 = np.ones(len(X1)).reshape(len(X1), 1)
    X = np.concatenate([X0, X1])
    C = np.concatenate([C0, C1])
    return X, C

def chargerFichiersManuscrits(formatFichiers, limit=0):
    '''
    Read the .mat data files corresponding to handwritten digits

    Parameters
    ----------
    formatFichiers : str
        filename format of the files to read, for example: 'DigitTrain_%d.mat'
    limit : int, optional
        Restrict each file to a certain number of entries. The default is 0 (no limit).

    Returns
    -------
    X : 
        data tensor.
    C : 
        expected output of the output layer of the neural net.
    '''
    all_X = []
    all_C = []
    for digit in range(10):
        Xi = lectureMatData(formatFichiers % digit, nomColonne='imgs')
        if limit > 0:
            Xi = Xi[:limit]
        Ci = np.zeros((len(Xi), 10))
        Ci[:, digit] = np.ones(len(Xi))
        all_X.append(Xi)
        all_C.append(Ci)
    return np.concatenate(all_X), np.concatenate(all_C)


def test(couche, X_test, C_test, mode='strict'):
    '''
    Test the neural net with test data and the expected output, and prints the success rate.

    Parameters
    ----------
    couche : neuralnet.layer.Layer
        Output layer.
    X_test : numpy array
        test data.
    C_test : numpy array
        expected output of the neural net.
    mode : str among ['strict','max']
        corresponds to how a success is evaluated
        in strict mode : a test successes if np.array_equal(output[i],C_test[i])
        in max mode : a test successes if np.argmax(output[i])==np.argmax(C_test[i])

    Returns
    -------
    n_succes : int
        number of successful predictions.

    '''
    assert C_test.shape == (len(X_test),len(couche))
    assert mode in ['strict','max']
    n_succes = 0
    res = couche(X_test)
    for i in range(len(res)):
        res[i] = (res[i] > 0.5)*1.0
        if (   (mode == 'strict' and np.array_equal(res[i], C_test[i])) 
            or (mode == 'max'    and np.argmax(res[i]) == np.argmax(C_test[i])) 
           ) :
            n_succes += 1
    print(100*n_succes/len(X_test), "% of success")
    return n_succes

def show(couche, X_test, C_test, n_succes, smooth=True):
    '''
    Plots a beautiful representation of the output of a neural network with 2D input

    Parameters
    ----------
    couche : neuralnet.layer.Layer
        The output layer of the neural network.
    X_test : tensor
        shape(n,2,1) (2D input).
    C_test : array
        shape (n, len(couche)).
    n_succes : int
        number of success (used for the title of the plot).
    smooth : bool, optional
        Smooth transition between class zones. The default is True.

    Returns
    -------
    None.
    '''
    plt.figure()
    fig = plt.gcf().number
    z, colors = graph.graph_2d_classes(
        couche,
        bounds=(
            np.amin(X_test),
            np.amax(X_test),
            np.amin(X_test),
            np.amax(X_test)),
        step=(np.amax(X_test) - np.amin(X_test))/100, fig=fig, smooth=smooth)
    graph.graph_2d_entry(X_test, C_test, colors, fig=fig)
    plt.title("{} succès sur {} données".format(n_succes, len(X_test)))
    plt.show()