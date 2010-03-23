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
    def f(x):
        return x
                   
    def df(x, y):
        return 1.

    f = staticmethod(f)
    df = staticmethod(df)

class TanhFunction:
    """
    Container class for the hyperbolic tangent function.
    
    Implements two static methods, one for the function and one for its derivatives.
    """
    def f(x):
        return np.tanh(x)
                   
    def df(x, y):
        return 1. - y**2
    
    f = staticmethod(f)
    df = staticmethod(df)
    
class LogisticFunction:
    """
    Container class for the logistic function.
    
    Implements two static methods, one for the function and one for its derivatives.
    """
    def f(x):
        return 1./(1. + np.exp(-x))
                   
    def df(x, y):
        return y * (1 - y)

    f = staticmethod(f)
    df = staticmethod(df)
        
class SoftmaxFunction: 
    """
    Container class for the soft-max function.
    
    Implements two static methods, one for the function and one for its derivatives.
    """
    def f(x):
        return np.exp(x) / sum(np.exp(x))
                   
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
        pass
    
    f = staticmethod(f)
    df = staticmethod(df)
    
class RBFunction: 
    """
    Container class for the radial basis function.
    
    Implements two static methods, one for the function and one for its derivatives.
    """
    def f(x):
        return np.exp(-(x*x))
                   
    def df(x, y):
        return -2*(x*y);
    
    f = staticmethod(f)
    df = staticmethod(df)
    
class SignFunction:
    """
    Container class for the sign function.
    
    Implements two static methods, one for the function and one for its derivatives.
    """
    def f(x):
        return np.sign(x)
                   
    def df(x, y):
        raise NotImplementedError, "Derivative of the sign function does not exist"

    f = staticmethod(f)
    df = staticmethod(df)

