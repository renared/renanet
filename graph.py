# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 23:19:49 2020

@author: Yann
"""

from colorsys import hsv_to_rgb, rgb_to_hsv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import renanet

def graph_2d_classes(output_layer, bounds=(0.0, 1.0, 0.0, 1.0), step=0.1, fig=-1, offline=False, smooth=True):
    '''    
    Parameters
    ----------
    output_layer : neuralnet.layer.Layer
        The output layer of a neural network with 2D points input.
    bounds : tuple (xmin,xmax,ymin,ymax)
        Bounds. The default is (0.0, 1.0, 0.0, 1.0).
    step : float, optional
        Resolution of the plot. The default is 0.1
    fig : int, optional
        Figure number, -1 (default) creates a new figure.
    offline : bool, optinal
        If True, only returns the image, doesn't plot anything.
    smooth : bool, optional
        If False, no smooth transition between colors.
    Returns
    -------
    plotted image and colors.
    '''
    assert type(output_layer) == renanet.layer.Layer
    #assert que la couche d'entrée prend un tenseur de taille (n,2,1)
    
    colors = np.array([
        hsv_to_rgb((i + 1)*0.7 / len(output_layer), 0.7, 0.9)
        for i in range(len(output_layer)+1)])   # il y a n_classes+1 couleurs, la couleur supplémentaire est la classe "autre" qui correspond à aucune des autres
    # xy = np.mgrid[bounds[0]:bounds[1]:step, bounds[2]:bounds[3]:step].reshape(2, -1).T
    x = np.arange(bounds[0], bounds[1], step)
    y = np.arange(bounds[2], bounds[3], step)
    xyi = np.array([(i, j) for i in range(len(x)) for j in range(len(y))])
    xy = np.array([(x[i], y[j]) for i, j in xyi])
    data = xy.reshape(*xy.shape, 1)

    output = output_layer(data)
    Z = output.reshape(len(x), len(y), len(output_layer))

    z = np.zeros((len(x), len(y), 3))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if smooth == False:
                Z[i,j] = (Z[i,j] > 0.5)   # [0,1] -> 0 ou 1
            weights = np.concatenate([
                [max(0, 1 - sum(Z[i, j]))],
                Z[i, j],
                ])   # poids des couleurs : on ajoute le poids de la classe "autre" max(0,1-sum(Z[i,j]))
            z[j, i] = np.average(colors, axis=0, weights=weights)  # barycentre des couleurs
            # pourquoi z[j,i] et pas z[i,j] ? honnêtement je ne sais pas, mais sinon ça n'affiche pas correctement !
            #z[i,j] = ((len(Z[i,j])-sum(Z[i,j]))*colors[0]+sum(Z[i,j]*colors[1:]))/len(Z[i,j])   # barycentre des couleurs
            
    if not offline:
        if fig >= 0:
            plt.figure(fig)
        else :
            plt.figure()
        plt.imshow(z, origin='lower', extent=bounds)
        
        # create a patch (proxy artist) for every color 
        patches = [
            mpatches.Patch(
                color=colors[i],
                label="Class {l}".format(l=i))
            for i in range(len(colors))
            ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        # if Z.shape[2]==1:
        #     plt.figure()
        #     plt.pcolormesh(x,y,Z.reshape(len(x),len(y)))
    return z, colors
    
def graph_2d_entry(tensor, classes, colors, fig=-1):
    '''
    Scatter plot of 2D input

    Parameters
    ----------
    tensor : 
        input tensor of shape (n, 2, 1).
    classes : 
        input tensor of shape (n, k) (k classes).
    colors : list
        List of colors, colors[0] is default, colors[1:] matches classes.
    fig : int, optional
        Figure number. -1 (default) creates a new figure.

    Returns
    -------
    None.
    '''
    assert len(tensor) > 0
    assert tensor[0].shape == (2, 1)
    assert len(tensor) == len(classes)
    assert len(colors) == classes.shape[1] + 1
    plt.figure(fig if fig >= 0 else None)
    classespoints = [[] for c in colors]
    # classespoints est une liste qui contient pour chaque couleur une liste des points de cette couleur
    n = len(tensor)
    for i in range(n):
        for j in range(len(classes[i])):
            if classes[i][j] == 1:
                classespoints[j+1].append(tensor[i].reshape(2))
                break
            # si aucune classe ne correspond
            classespoints[0].append(tensor[i].reshape(2))
    for j in range(len(classespoints)):
        plt.scatter(*zip(*classespoints[j]), color=colors[j], marker='X', edgecolors='black')
    