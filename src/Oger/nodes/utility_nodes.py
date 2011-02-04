import mdp.nodes
import scipy.signal
import numpy as np

class FeedbackNode(mdp.Node):
    """FeedbackNode creates the ability to feed back a certain part of a flow as 
    input to the flow. It both implements the Node API and the generator API and
    can thus be used as input for a flow.
    
    The duration that the feedback node feeds back data can be given. Prior to using 
    the node as data generator, it should be executed so it can store the previous 
    state.
    
    When a FeedbackNode is reused several times, reset() should be called prior to
    each use which resets the internal counter.
    
    Note that this node keeps state and can thus NOT be used in parallel using threads.
    """ 
    def __init__(self, n_timesteps=1, input_dim=None, dtype=None):
        super(FeedbackNode, self).__init__(input_dim=input_dim, output_dim=input_dim, dtype=dtype)
        
        self.n_timesteps = n_timesteps
        self.last_value = None
        self.current_timestep = 0
        
    def reset(self):
        self.current_timestep = 0
        
    def is_trainable(self):
        return True
    
    def _train(self, x, y):
        self.last_value = mdp.numx.atleast_2d(y[-1, :])

    def __iter__(self):
        while self.current_timestep < self.n_timesteps:
            self.current_timestep += 1
            
            yield self.last_value
                
    def _execute(self, x):
        self.last_value = mdp.numx.atleast_2d(x[-1, :])
        return x

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
        e = mdp.numx.atleast_2d(mdp.numx.mean(x, axis=0, dtype=self.dtype))
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
        max_indices = mdp.numx.argmax(x, axis=1)
        r = -mdp.numx.ones_like(x)
        for i in range(r.shape[0]):
            r[i, max_indices[i]] = 1
        return r
    
    
class ShiftNode(mdp.Node):
    """Return input data shifted one or more time steps.

    This is useful for architectures in which data from different time steps is
    needed. The values that are left over are set to zero.

    Negative shift values cause a shift back in time and positive ones forward in time.
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
        
        
class ResampleNode(mdp.Node):
    """ Resamples the input signal. Based on scipy.signal.resample
    
    CODE FROM: Georg Holzmann
    """
    
    def __init__(self, input_dim=None, ratio=0.5, dtype='float64', window=None):
        """ Initializes and constructs a random reservoir.
                
        input_dim -- the number of inputs (output dimension is always
                     the same as input dimension)
    
        ratio     -- ratio of up or down sampling
                     (e.g. 0.5 means downsampling to half the samplingrate)
        
        window    -- see window parameter in scipy.signal.resample
        """
        super(ResampleNode, self).__init__(input_dim, input_dim, dtype)
        self.ratio = ratio
        self.window = window
        
    def is_trainable(self):
        return False
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Resample input vector x.
        """
        self.oldlength = len(x)
        newlength = self.oldlength * self.ratio
        sig = scipy.signal.resample(x, newlength, window=self.window)
        return sig.copy()
    
    def _inverse(self, y):
        """ Inverse the resampling.
        """
        sig = scipy.signal.resample(y, self.oldlength, window=self.window)
        return sig.copy()
        
class TimeFramesNode2(mdp.nodes.TimeFramesNode):
    """ An extension of TimeFramesNode that preserves the temporal
    length of the data.
    """
    def __init__(self, time_frames, input_dim=None, dtype=None):
        super(TimeFramesNode2, self).__init__(input_dim=input_dim, dtype=dtype, time_frames=time_frames)

    def _execute(self, x):
        tf = x.shape[0] - (self.time_frames - 1)
        rows = self.input_dim
        cols = self.output_dim
        y = mdp.numx.zeros((x.shape[0], cols), dtype=self.dtype)
        for frame in range(self.time_frames):
            y[-tf:, frame * rows:(frame + 1) * rows] = x[frame:frame + tf, :]
        return y
    
    def pseudo_inverse(self, y):
        pass
        
        
class FeedbackShiftNode(mdp.Node):
    """ Shift node that can be applied when using generators. 
    The node works as a delay line with the number of timesteps the lengths of the delay line.
    """

    def __init__(self, input_dim=None, output_dim=None, n_shifts=1,
                 dtype='float64'):
        super(FeedbackShiftNode, self).__init__(input_dim, output_dim, dtype)
        self.n_shifts = n_shifts
        self.y = None
    def is_trainable(self):
        False

    def _execute(self, x):
        n = x.shape
        assert(n > 1)

        if self.y == None:
            self.y = np.zeros((self.n_shifts,self._input_dim))

        self.y = np.vstack([self.y,x.copy()])

        returny = self.y[:x.shape[0],:].copy()
        self.y = self.y[x.shape[0]:,:]
        return returny

    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = n
        

class RescaleZMUSNode(mdp.Node):
    '''
    Rescales the output with the mean and standard deviation seen during training
    
    If 'use_var' is set, the variance is used instead of the standard deviation
    
    Currently for 1 input only!!
    '''
    def __init__(self, use_var=False, input_dim =1, dtype = None):
        super(RescaleZMUSNode,self).__init__(input_dim=input_dim, dtype=dtype)
        self._mean = 0
        self._std = 0
        self._len = 0
        self._use_var = use_var
    
    def is_trainable(self):
        return True
    
    def _train(self,x):
        self._mean += mdp.numx.mean(x) * len(x)
        self._std += mdp.numx.sum(x**2) - mdp.numx.sum(x)**2
        self._len += len(x)
        
    def _stop_training(self):
        self._mean /= self._len
        self._std /= self._len
        if self._use_var:
            self._std = mdp.numx.sqrt(self._std)
    
    def _execute(self, x):
        return (x-self._mean) / self._std
    
    