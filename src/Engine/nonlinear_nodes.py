'''
Created on Aug 24, 2009

@author: dvrstrae
'''
import numpy as np
import mdp

class SignNode(mdp.Node):
    """
     Compute the sign function of the input.
        
        This simple node computes the sign function of its input
    
    # PyUML: Do not remove this line! # XMI_ID:__6oqMK0EEd6_mKvLwRcUQA
    """
    
    def __init__(self, input_dim=None, output_dim=None, dtype='float64'):
        
        super(SignNode, self).__init__(input_dim, output_dim, dtype)
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        return np.sign(x)
