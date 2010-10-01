import Oger
import mdp
from mdp import numx
from mdp.utils import mult

# TODO: could we turn this in a generic "function" node?
class ThresholdNode(mdp.Node):
    """
    Compute a threshold function of the input.
    """
    
    def __init__(self, input_dim=None, output_dim=None, threshold=0, dtype='float64'):
        
        super(ThresholdNode, self).__init__(input_dim, output_dim, dtype)
        self.threshold = threshold
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        #print self.threshold
        return (x > self.threshold) - 1 * (x < self.threshold) 


# TODO what about the mdp.PerceptronClassifier??
class PerceptronNode(mdp.Node):
    """
    Trains a (non-linear) perceptron using gradient descent.

    The output transfer function can be specified together with the appropriate
    derivative for the input (only needed for back propagation). 

    Using softmax as the transfer function turns this into logistic regression.
    """

    # TODO: re-include Oger.utils.LinearFunction as default value for transfer_func 
    def __init__(self, input_dim=None, output_dim=None, dtype='float64', transfer_func=Oger.utils.LinearFunction):        
        super(PerceptronNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        
        self.transfer_func = transfer_func

        # TODO: it would be nice if the dimensions could be derived automatically:
        self.w = self._refcast(mdp.numx.random.randn(self.input_dim, self.output_dim) * 0.01)
        self.b = self._refcast(mdp.numx.random.randn(self.output_dim) * 0.01)

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        n, d = x.shape
        if n > 1:
            bias = numx.tile(self.b, (n, 1))
        else:
            bias = self.b
        y = self.transfer_func.f(mult(x, self.w) + bias)
        return y

    def is_trainable(self):
        return False

