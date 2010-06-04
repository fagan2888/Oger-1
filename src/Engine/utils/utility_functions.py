import mdp
import itertools

def get_spectral_radius(W):
    """
    get_spectral_radius(W)-> spec_rad
    Return the spectral radius (largest absolute eigenvalue) of the matrix W
    
    """
    return mdp.numx.amax(mdp.numx.absolute(mdp.numx.linalg.eigvals(W))) 

def empty_n_d_list(dims):
    ''' Create an n-dimensional list (n = len(dims)), of sizes given by the tuple dims
        E.g. Engine.utils.empty_n_d_list((3,1,2)) returns
        [[[[], []]], [[[], []]], [[[], []]]]
    '''
    if len(dims) > 1:
        return [empty_n_d_list(dims[1:]) for _ in range(dims[0])]
    else:
        return [[] for _ in range(dims[0])]

class ConcatenatingIterator(object):
    ''' Return an iterator which concatenates at every timestep its separate inputs to one large vector.
        It is useful for cases where you need to combine a FeedbackNode (which is an iterator) with some external input 
        (which is usually a Numpy array). However, it can be used to arbitrarily combine iterators and numpy arrays (or 
        other iterables).
        For a usage example, see tunable_sinewave.py under the /examples directory.
    '''
    def __init__(self, x, y=None):
        # If all inputs are arrays, we don't need to iterate over them, we can just return an array
        if mdp.numx.all([type(x_i) == mdp.numx.ndarray for x_i in x]):
            self.iterator_x = [x].__iter__()
        else:
            self.iterator_x = itertools.izip(*[iterable.__iter__() for iterable in x])
        
        # If all inputs are arrays, we can just concatenate them
        if y is not None:
            if mdp.numx.all([type(x_i) == mdp.numx.ndarray for x_i in x]) and type(y) == mdp.numx.ndarray:
                self.iterator_y = [y].__iter__()
            else:
                self.iterator_y = y.__iter__() 
        else:
            self.iterator_y = None    

    def __iter__(self):
        while True:
            x = mdp.numx.atleast_2d(mdp.numx.hstack(map(mdp.numx.atleast_2d, self.iterator_x.next())))
            
            if self.iterator_y is not None:
                yield (x, mdp.numx.atleast_2d(self.iterator_y.next()))
            else:
                yield x 
                                                        

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
