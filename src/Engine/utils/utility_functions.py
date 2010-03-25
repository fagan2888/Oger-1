'''
Created on Aug 20, 2009

@author: dvrstrae
'''
import mdp

def get_spectral_radius(W):
    return mdp.numx.amax(mdp.numx.absolute(mdp.numx.linalg.eigvals(W))) 

class LinearFunction:
    """
    Container class for the linear function.
    
    Implements two static methods, one for the function and one for its
    derivatives.
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
    
    Implements two static methods, one for the function and one for its
    derivatives.
    """
    def f(x):
        return mdp.numx.tanh(x)
                   
    def df(x, y):
        return 1. - y ** 2
    
    f = staticmethod(f)
    df = staticmethod(df)
    
class LogisticFunction:
    """
    Container class for the logistic function.
    
    Implements two static methods, one for the function and one for its
    derivatives.
    """
    def f(x):
        return 1. / (1. + mdp.numx.exp(-x))
                   
    def df(x, y):
        return y * (1 - y)

    f = staticmethod(f)
    df = staticmethod(df)
        
class SoftmaxFunction: 
    """
    Container class for the soft-max function.
    
    Implements two static methods, one for the function and one for its
    derivatives.
    """
    def f(x):
        n, d = x.shape
        out = mdp.numx.zeros((n, d))
        for i in range(n):
            activation = mdp.numx.exp(x[i, :])
            out[i, :] = activation / mdp.numx.sum(activation)
        return out
                   
    def df(x, y):
#        [Sm,Q] = size(y);
#        da_dn = cell(1,Q);
#        for q=1:Q
#            da_dn{q} = diag(y(:,q))*sum(y(:,q)) - kron(y(:,q)',y(:,q));
#
#        d = numx.zeros_like(y)
#        for i in range(y.shape[1]):
#          numx.diag(y[:,i])*sum(y[:,i]) - numx.kron(y[:,i:i+1].T, y[:,i:i+1]);
#            
#        raise d
        # TODO: In most cases we want the derivative with respect to the error
        # instead of the explicit derivative. For now I set it to 1. to not
        # break the perceptron but this should obviously be dealt with in a
        # more elegant way.
        return 1.
    
    f = staticmethod(f)
    df = staticmethod(df)
    
class RBFunction: 
    """
    Container class for the radial basis function.
    
    Implements two static methods, one for the function and one for its
    derivatives.
    """
    def f(x):
        return mdp.numx.exp(-(x * x))
                   
    def df(x, y):
        return - 2 * (x * y);
    
    f = staticmethod(f)
    df = staticmethod(df)
    
class SignFunction:
    """
    Container class for the sign function.
    
    Implements two static methods, one for the function and one for its
    derivatives.
    """
    def f(x):
        return mdp.numx.sign(x)
                   
    def df(x, y):
        raise NotImplementedError, "Derivative of the sign function does not exist"

    f = staticmethod(f)
    df = staticmethod(df)

