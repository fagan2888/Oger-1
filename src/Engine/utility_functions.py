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
#        [Sm,Q] = size(y);
#        da_dn = cell(1,Q);
#        for q=1:Q
#            da_dn{q} = diag(y(:,q))*sum(y(:,q)) - kron(y(:,q)',y(:,q));

#        d = np.zeros_like(y)
#        for i in range(y.shape[1]):
#          np.diag(y[:,i])*sum(y[:,i]) - np.kron(y[:,i:i+1].T, y[:,i:i+1]);
#            
#        raise d
        raise NotImplementedError, "TODO"
    
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

