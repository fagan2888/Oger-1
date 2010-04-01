import mdp

def get_spectral_radius(W):
    """
    get_spectral_radius(W)-> spec_rad
    Return the spectral radius (largest absolute eigenvalue) of the matrix W
    
    """
    return mdp.numx.amax(mdp.numx.absolute(mdp.numx.linalg.eigvals(W))) 

class LinearFunction:
    """
    Container class for the linear function.
    
    Implements two static methods, one for the function and one for its
    derivatives.
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
    
    Implements two static methods, one for the function and one for its
    derivatives.
    """
    @staticmethod
    def f(x):
        return mdp.numx.tanh(x)
                   
    @staticmethod
    def df(x, y):
        return 1. - y ** 2
    
class LogisticFunction:
    """
    Container class for the logistic function.
    
    Implements two static methods, one for the function and one for its
    derivatives.
    """
    @staticmethod
    def f(x):
        return 1. / (1. + mdp.numx.exp(-x))
                   
    @staticmethod
    def df(x, y):
        return y * (1 - y)
        
class SoftmaxFunction: 
    """
    Container class for the soft-max function.
    
    Implements two static methods, one for the function and one for its
    derivatives.
    """
    @staticmethod
    def f(x):
        n, d = x.shape
        out = mdp.numx.zeros((n, d))
        for i in range(n):
            activation = mdp.numx.exp(x[i, :])
            out[i, :] = activation / mdp.numx.sum(activation)
        return out
                   
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
    
    Implements two static methods, one for the function and one for its
    derivatives.
    """
    @staticmethod
    def f(x):
        return mdp.numx.sign(x)
                   
    @staticmethod
    def df(x, y):
        raise NotImplementedError, "Derivative of the sign function does not exist"
