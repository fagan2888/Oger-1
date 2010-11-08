import Oger
import mdp
from mdp import numx as numx
import numpy
from numpy import dot as dot
from mdp.utils import mult
import scipy

# TODO: could we turn this in a generic "function" node?
class ThresholdNode(mdp.Node):
    """
    Compute a threshold function of the input.
    Returns -1 or 1
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
        return (x > self.threshold) - 1 * (x <= self.threshold) 


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


class _LogisticRegressionBaseNode (mdp.Node):
    """ 
    Abstract node implementing all common stuff for Logistic Regression.
    
    Subclasses need to implement at least the _train(self,x,y) and _stop_training() methods!
    """

    def __init__(self,  input_dim=None, output_dim=None, dtype=None):
        super(_LogisticRegressionBaseNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self._x, self._y = None, None
    
    def is_invertible(self):
        return False

    def _train(self, x, y):
        # Make sure x and labels have the save dimensions
        if y.shape[0] != x.shape[0]:
            msg = ("The number of labels should be equal to the number of datapoints (%d != %d)" % (y.shape[0], x.shape[0]))
            raise mdp.TrainingException(msg)
        if self._x == None:
            self._x = mdp.numx.concatenate((x, mdp.numx.ones((x.shape[0],1))), axis=1)
            self._y = y
        else:
            self._x = mdp.numx.concatenate((self._x, mdp.numx.concatenate((x, mdp.numx.ones((x.shape[0],1))), axis=1)))
            self._y = mdp.numx.concatenate((self._y, y))

    def _execute(self, x):
        return Oger.utils.LogisticFunction.f(dot(x, self.w) + self.b)


class IRLSLogisticRegressionNode (_LogisticRegressionBaseNode):
    """ 
    Node implementing logistic regression with Iterative Reweighted Least Squares (IRLS).
    
    Regularisation is often needed to allow conversion, avoid singular matrices
    and limit calculation time!
    Training of this node can be very memory intensive because all train data is saved and processed later
    """
    
    def __init__(self, regul_param=0, epochs=20, threshold=0.0001, use_pinv=False, input_dim=None, output_dim=None, dtype=None):
        super(IRLSLogisticRegressionNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.regul_param = regul_param
        self._threshold = threshold
        self._epochs = epochs
        self.use_pinv = use_pinv

    def _stop_training(self):
        if self.use_pinv:
            inv = mdp.utils.pinv
        else:
            inv = mdp.utils.inv
        
        if mdp.numx.rank(self._y)==1:
            self._y = self._y.reshape((len(self._y),1))
            
        w = dot(dot(inv(dot(self._x.T, self._x) + self.regul_param * numpy.eye(self._x.shape[1])),self._x.T), self._y)
        
        for i in range(self._epochs):
            w_old = w
            yn = Oger.utils.LogisticFunction.f(dot(self._x, w)).reshape(self._y.shape)
            nyn = 1-yn
            # Avoid log(0)
            epsilon = numpy.float(numpy.finfo(self._x.dtype).tiny)
            yn[yn==0] = epsilon
            nyn[nyn==0] = epsilon
            
            z = dot(self._x, w).reshape(self._y.shape) - (yn - self._y) / yn / nyn
            R = dot(yn * nyn, mdp.numx.ones((1,self._x.shape[1]))).T
            w = dot(dot(inv(dot(self._x.T * R, self._x) + self.regul_param * numpy.eye(self._x.shape[1])),self._x.T) * R, z)
            
            if numpy.mean(abs(w - w_old)) < self._threshold:
                break
        
        if i == self._epochs - 1 and not numpy.mean(abs(w - w_old)) < self._threshold:
            str = 'Algorithm did not converge in %d steps. Try a larger regularisation parameter.' % self._epochs
            print str
        
        self.w = w[0:-1]
        self.b = w[-1]
        

class LogisticRegressionNode (_LogisticRegressionBaseNode):
    """ 
    Node implementing logistic regression
    
    Examples of gradient training methods (gtrainer):
        Oger.gradient.CGTrainer(), 
        Oger.gradient.BFGSTrainer(),
        Oger.gradient.GradientDescentTrainer(epochs=30000),
        Oger.gradient.RPROPTrainer(epochs=30000), 
        Oger.gradient.LBFGSBTrainer(weight_bounds=10)
    Parameters of the gtrainer that need to be optimized can be set using the gtrainer_arg and 
    gtrainer_arg_val variables
    
    Training of this node can be very memory intensive because all train data is saved and processed later
    Some trainers only support single output
    """
    
    def __init__(self, gtrainer=None, gtrainer_arg=None, gtrainer_arg_val=None, input_dim=None, output_dim=None, dtype=None):
        super(LogisticRegressionNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self._gtrainer = gtrainer
        self._gtrainer_arg = gtrainer_arg
        self._gtrainer_arg_val = gtrainer_arg_val
    
    def _stop_training(self):
        # initialize weight vector
        w = mdp.numx.rand(self._x.shape[1]) * 2 - 1
#        w *= numpy.min(numpy.abs(self._x))
#        w = dot(dot(mdp.utils.inv(dot(self._x.T, self._x)),self._x.T), self._y)
        
        
        # solves most of the vector vs. array problems 
        if numpy.rank(w)>1:
            w_shape = w.shape
        else:
            w_shape = numpy.array(numpy.mat(w).T).shape
        
        loss = lambda w : Oger.utils.ce(Oger.utils.LogisticFunction.f(dot(self._x, w)).reshape(self._y.shape), self._y)
        d_loss = lambda w : dot((Oger.utils.LogisticFunction.f(dot(self._x, w)).reshape(self._y.shape) - self._y).T, self._x).reshape(w.shape)
        func = lambda w : (d_loss(w), loss(w))
        
        if self._gtrainer_arg:
            setattr(self._gtrainer, self._gtrainer_arg, self._gtrainer_arg_val)
        
        w = self._gtrainer.train(func, w).reshape(w_shape)
        
        self.w = w[0:-1]
        self.b = w[-1]