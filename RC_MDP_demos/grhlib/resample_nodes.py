#===============================================================================
# Resample Node for the MDP package
#===============================================================================

import mdp
import scipy.signal


class ResampleNode(mdp.Node):
    """ Resamples the input signal. Based on scipy.signal.resample
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



#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    from numpy import *
    import pylab, sys, errormeasures
    sys.path.append("../Experimentfase1")
    import testsignals
    
    size = 983
    x = zeros((size,2))
    x[:,0] = testsignals.sine(size,0.0943,0.3) \
             + testsignals.rectangular(size,0.173,0.3) \
             + testsignals.triangular(size,0.1193,0.3)
    x[:,1] = testsignals.rectangular(size,0.01)
    
    resampler = ResampleNode(2, 0.56, window="hamming")
#    resampler = ResampleNode(2, 0.5)
    y = resampler(x)
    x2 = resampler.inverse(y)
    print x.shape, y.shape, x2.shape
    
    # calc MSE
    print "MSE first signal: ", errormeasures.mse(x[:,0],x2[:,0])
    print "MSE second signal: ", errormeasures.mse(x[:,1],x2[:,1])
    
    pylab.subplot(2,1,1)
    pylab.plot(x[:,0])
    pylab.plot(x2[:,0],'r')
    pylab.subplot(2,1,2)
    pylab.plot(x[:,1])
    pylab.plot(x2[:,1],'r')
    pylab.show()

