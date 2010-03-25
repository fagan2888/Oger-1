'''
Created on Aug 24, 2009

@author: dvrstrae
'''
import numpy as np
import Engine
import mdp
from mdp.utils import mult

# TODO: is this node ever used?
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


class PerceptronNode(mdp.Node):
    """
    Trains a (non-linear) perceptron using gradient descent.

    The output transfer function can be specified together with the appropriate
    derivative for the input (only needed for back propagation). 

    Using softmax as the transfer function turns this into logistic regression.
    """

    # TODO: re-include Engine.utils.LinearFunction as default value for transfer_func 
    def __init__(self, transfer_func, input_dim=None, output_dim=None, dtype='float64'):        
        super(PerceptronNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        
        self.transfer_func = transfer_func

        # TODO: it would be nice if the dimensions could be derived automatically:
        self.w = self._refcast(mdp.numx.random.randn(self.input_dim, self.output_dim) * 0.01)
        self.b = self._refcast(mdp.numx.random.randn(self.output_dim) * 0.01)

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        n, d = x.shape
        if n > 1:
            bias = np.tile(self.b, (n, 1))
        else:
            bias = self.b
        y = self.transfer_func.f(mult(x, self.w) + bias)
        return y

    def is_training(self):
        return False

    def is_trainable(self):
        return False

