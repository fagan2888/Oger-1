'''
Created on Aug 20, 2009

@author: dvrstrae
'''
import numpy as np


def get_spectral_radius(W):
    return np.max(np.abs(np.linalg.eigvals(W))) 

def logistic(x):
    return 1./(1. + np.exp(-x))

def logistic_d(x):
    y = logistic(x)
    return y * (1 - y)
