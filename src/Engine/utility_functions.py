'''
Created on Aug 20, 2009

@author: dvrstrae
'''
import numpy as np


def get_specrad(W):
    return np.max(np.abs(np.linalg.eigvals(W))) 