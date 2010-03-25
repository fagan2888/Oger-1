'''
Created on Aug 20, 2009

@author: dvrstrae
'''
import mdp

class FeedbackNode(mdp.Node):
    def __init__(self, n_timesteps=1, input_dim=None, output_dim=None, dtype=None):
        super(FeedbackNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.current_timestep = 0
        self.n_timesteps = n_timesteps
        self.last_value = None
        
    def is_trainable(self):
        return True

    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = n
        
    def __iter__(self):
        while self.current_timestep < self.n_timesteps:
            self.current_timestep += 1
            yield self.last_value
            
    def _train(self, x):
        self.last_value = mdp.numx.atleast_2d(x[-1, :])
                
    def _execute(self, x):
        self.last_value = x
        return self.last_value

class WashoutNode(mdp.Node):
    """
         Remove initial states.
    
     """

    def __init__(self, washout, input_dim=None, dtype='float64'):
        super(WashoutNode, self).__init__(input_dim, input_dim, dtype)
        self.washout = washout

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        return x[self.washout:,:]

class MeanAcrossTimeNode(mdp.Node):
    """
        Compute mean across time (2nd dimension)
        
    """

    def __init__(self, input_dim=None, output_dim=None, dtype='float64'):
        super(MeanAcrossTimeNode, self).__init__(input_dim, output_dim, dtype)
        
    def is_trainable(self):
        return False

    def is_invertible(self):
        return False
    
    def _check_train_args(self, x, y):
        # set output_dim if necessary
        if self._output_dim is None:
            self._set_output_dim(y.shape[1])

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        e=mdp.numx.mean(x, axis=0, dtype=self.dtype)
        return e

class WTANode(mdp.Node):
    """
        Compute Winner take-all at every timestep (2nd dimension)
        
    """

    def __init__(self, input_dim=None, output_dim=None, dtype='float64'):
        super(WTANode, self).__init__(input_dim, output_dim, dtype)
        
    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _check_train_args(self, x, y):
        #set output_dim if necessary
        if self._output_dim is None:
            self._set_output_dim(y.shape[1])
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        max_indices= mdp.numx.argmax(x, axis=1)
        r=-mdp.numx.ones_like(x)
        for i in range(r.shape[0]):
            r[i,max_indices[i]] = 1
        return r
    
    
class ShiftNode(mdp.Node):
    """Return input data shifted one or more time steps.

    This is useful for architectures in which data from different time steps is
    needed. The values that are left over are set to zero.

    Negative shift values cause a shift to the left and positive ones to the
    right.
    """

    def __init__(self, input_dim=None, output_dim=None, n_shifts=1,
                 dtype='float64'):
        super(ShiftNode, self).__init__(input_dim, output_dim, dtype)
        self.n_shifts = n_shifts

    def is_trainable(self):
        False

    def _execute(self, x):
        n = x.shape
        assert(n > 1)

        ns = self.n_shifts
        y = x.copy()

        if ns < 0:
            y[:ns] = x[-ns:]
            y[ns:] = 0
        elif ns > 0:
            y[ns:] = x[:-ns]
            y[:ns] = 0
        else:
            y = x

        return y

    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = n