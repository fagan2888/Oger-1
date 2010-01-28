'''
Created on Aug 20, 2009

@author: dvrstrae
'''
import mdp

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