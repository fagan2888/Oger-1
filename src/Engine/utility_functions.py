'''
Created on Aug 20, 2009

@author: dvrstrae
'''
import numpy as np


def get_spectral_radius(W):
    return np.max(np.abs(np.linalg.eigvals(W))) 

class LinearFunction:
    """
    Container class for the linear function.
    
    Implements two static methods, one for the function and one for its derivatives.
    """
    @staticmethod
    def f(x):
        return x
                   
    @staticmethod
    def df(x, y):
        return 1.

class TanhFunction:
    """
    Container class for the hyperbolic tangent function.
    
    Implements two static methods, one for the function and one for its derivatives.
    """
    @staticmethod
    def f(x):
        return np.tanh(x)
                   
    @staticmethod
    def df(x, y):
        return 1. - y**2
    
class LogisticFunction:
    """
    Container class for the logistic function.
    
    Implements two static methods, one for the function and one for its derivatives.
    """
    @staticmethod
    def f(x):
        return 1./(1. + np.exp(-x))
                   
    @staticmethod
    def df(x, y):
        return y * (1 - y)
        
class SoftmaxFunction: 
    """
    Container class for the soft-max function.
    
    Implements two static methods, one for the function and one for its derivatives.
    """
    @staticmethod
    def f(x):
        return np.exp(x) / sum(np.exp(x))
                   
    @staticmethod
    def df(x, y):
        # TODO: In most cases we want the derivative with respect to the error
        # instead of the explicit derivative. For now I set it to 1. to not
        # break the perceptron but this should obviously be dealt with in a
        # more elegant way.
        return 1.
        
class SignFunction:
    """
    Container class for the sign function.
    
    Implements two static methods, one for the function and one for its derivatives.
    """
    @staticmethod
    def f(x):
        return np.sign(x)
                   
    @staticmethod
    def df(x, y):
        raise NotImplementedError, "Derivative of the sign function does not exist"

<<<<<<< local
=======
    f = staticmethod(f)
    df = staticmethod(df)

class CrossEntropyFunction:
    """
    Container class for the multi-class cross-entropy function.

    Implements two static methods, one for the function and one for its
    derivatives.
    """
    def f(x, t):
        return sum(-numx.log(x[t == 1]))

    def df(x, t):
        return -t/x

    f = staticmethod(f)
    df = staticmethod(df)
>>>>>>> other
