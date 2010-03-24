"""
A collection of some trainers that use gradient information.

The convention is to minimize the objective (error) function.
"""

from mdp import numx
import scipy.optimize as opt

# More can be found here:
#  - http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
#  - http://openopt.org/Welcome
#  - http://wiki.sagemath.org/optimization

class CGTrainer:
    def train(self, func, x0):
        fobj = lambda x: func(x)[1]
        fprime = lambda x: func(x)[0]
        return opt.fmin_cg(fobj, x0, fprime)

class BFGSTrainer:
    def train(self, func, x0):
        fobj = lambda x: func(x)[1]
        fprime = lambda x: func(x)[0]
        return opt.fmin_bfgs(fobj, x0, fprime, disp=2)
        
class LBFGSBTrainer:
    def __init__(self, weight_bounds=(-1,1)):
        self.weight_bounds = weight_bounds
        
    def train(self, func, x0):
        fobj = lambda x: func(x)[1]
        fprime = lambda x: func(x)[0]
        bounds = [self.weight_bounds,] * x0.size
        return opt.fmin_l_bfgs_b(fobj, x0, fprime=fprime, bounds=bounds)[0]

class GradientDescentTrainer:
    def __init__(self, learning_rate=.01, momentum=0, epochs=1, decay=0):
        """
            learning_rate -- size of the gradient steps (default = .001)
            
            momentum -- momentum term (default = 0)
    
            epochs -- number of times to do updates on the same data (default = 1)
    
            decay -- weight decay term (default = 0)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.decay = decay
        
        self.dparams = None
        
    def train(self, func, params):
        """
        Perform gradient descent over params using the gradient returned by func.
        """
    
        if self.dparams is None:
            self.dparams = numx.zeros(params.shape)
            
        updated_params = params
    
        for _ in range(self.epochs):
            gradient = func(updated_params)[0]
            self.dparams = self.momentum * self.dparams - self.learning_rate * gradient
            # TODO: how do we make sure that we do not decay the bias terms?
            updated_params += self.dparams - self.decay * updated_params
    
        return updated_params

class RPROPTrainer:
    def __init__(self, etamin=0.5, etaplus=1.2, deltamin=10e-6, deltamax=50, deltainit=0.0125, epochs=1):

        self._uW = None
        self.deltaW = None

        self.etamin = etamin;
        self.etaplus = etaplus;
        self.deltamin = deltamin;
        self.deltamax = deltamax;
        self.epochs = epochs
        self.deltainit = deltainit

    def train(self, func, params):
        if self._uW is None:
            # TODO: should we not refcast here?
            self._uW = numx.zeros_like(params)
            self.deltaW = numx.ones_like(params)*self.deltainit
            
        updated_params = params.copy()
    
        for _ in range(self.epochs):
            # TODO: properly name variables
            uW = func(updated_params)[0]

            WW = self._uW * uW;

            self.deltaW *= self.etaplus*(WW>0)+self.etamin*(WW<0)+1*(WW==0);
        
            self.deltaW = numx.maximum(self.deltaW, self.deltamin)
            self.deltaW = numx.minimum(self.deltaW, self.deltamax)
        
            updated_params -= self.deltaW*numx.sign(uW)
    
            self._uW = uW

        return updated_params

