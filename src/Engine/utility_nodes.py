'''
Created on Aug 20, 2009

@author: dvrstrae
'''
import mdp
import scipy as sp

class WashoutNode(mdp.Node):
    """
     remove initial states.
    
    # PyUML: Do not remove this line! # XMI_ID:__6v-8K0EEd6_mKvLwRcUQA
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
        
    # PyUML: Do not remove this line! # XMI_ID:__6v-8K0EEd6_mKvLwRcUQA
    """

    def __init__(self, input_dim=None, dtype='float64'):
        super(MeanAcrossTimeNode, self).__init__(input_dim, input_dim, dtype)
        
    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        return sp.mean(x, axis=0, dtype=self.dtype)

class WTANode(mdp.Node):
    """
        Compute Winner take-all at every timestep (2nd dimension)
        
    # PyUML: Do not remove this line! # XMI_ID:__6v-8K0EEd6_mKvLwRcUQA
    """

    def __init__(self, input_dim=None, dtype='float64'):
        super(WTANode, self).__init__(input_dim, input_dim, dtype)
        
    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        max_indices=sp.argmax(x, axis=1)
        r=-sp.ones_like(x)
        for i in range(r.shape[0]):
            r[i,max_indices[i]] = 1
        return r