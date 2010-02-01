'''
Created on Aug 20, 2009

@author: dvrstrae
'''
from utility_functions import get_specrad
import mdp
import numpy

class ReservoirNode(mdp.Node):
    """
     Very experimental demo !
    
    # PyUML: Do not remove this line! # XMI_ID:__6rGcK0EEd6_mKvLwRcUQA
    """
    
    def __init__(self, input_dim=None, output_dim=None, spec_radius=0.9, 
                 nonlin_func = 'tanh', bias = 0, input_scaling=1, dtype='float64'):
        """ Initializes and constructs a random reservoir.
                
        output_dim -- the number of outputs, which is also the number of
                          neurons in the reservoir
        prototype -- a prototype reservoir which will be cloned with all
                     its parameters
        """
        super(ReservoirNode, self).__init__(input_dim, output_dim, dtype)
        
        self.input_scaling = input_scaling
        self.Win = self.input_scaling*(numpy.random.randint(0,2, [output_dim, input_dim])*2-1) #, output_dim))
        self.bias = bias
        self.Biasin = numpy.ones(output_dim)*self.bias
        self.W = mdp.numx_rand.randn(output_dim,output_dim)

        # scale it to spectral radius
        self.W *= spec_radius/get_specrad(self.W)
        
        # make a hook object (demo)
        self.nonlin_func=nonlin_func
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        
        nonlinearity = getattr(mdp.numx, self.nonlin_func)
        steps = x.shape[0]
        self.states = numpy.zeros((steps, self._output_dim), dtype=self._dtype)
        self.state = numpy.zeros(self._output_dim)
        self.state_nonlin = numpy.zeros(self._output_dim)

        for n in range(steps):
            self.state = numpy.dot(self.W, self.state)
            self.state += numpy.dot(self.Win, x[n])
            self.state += self.Biasin
            self.state_nonlin = nonlinearity(self.state)
            # call the hook
            #self.hook.execute()
            
            self.states[n] = self.state_nonlin
        
        return self.states


class LeakyReservoirNode(ReservoirNode):

    def __init__(self, input_dim=None, output_dim=None, spec_radius=0.9, 
                 nonlin_func = 'tanh', bias = 0.0, input_scaling=1.0, leak_rate=1.0, dtype='float64'):
        """ Initializes and constructs a random reservoir with leaky-integrator neurons.
                
        output_dim -- the number of outputs, which is also the number of
                          neurons in the reservoir
        prototype -- a prototype reservoir which will be cloned with all
                     its parameters
        """
        super(LeakyReservoirNode, self).__init__(input_dim, output_dim, spec_radius,
                                            nonlin_func, bias, input_scaling, dtype)
        
        self.leak_rate = leak_rate

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        
        nonlinearity = getattr(mdp.numx, self.nonlin_func)
        steps = x.shape[0]
        self.states = numpy.zeros((steps, self._output_dim), dtype=self._dtype)
        self.state = numpy.zeros(self._output_dim)
        self.state_nonlin = numpy.zeros(self._output_dim)
        leak_rate = self.leak_rate

        self.state += numpy.dot(self.Win, x[0])
        self.state += self.Biasin
        self.state_nonlin = nonlinearity(self.state)
        self.states[0] = leak_rate * self.state_nonlin

        for n in range(1, steps):
            self.state = numpy.dot(self.W, self.state)
            self.state += numpy.dot(self.Win, x[n])
            self.state += self.Biasin
            self.state_nonlin = nonlinearity(self.state)
            
            self.states[n] = (1 - leak_rate) * self.states[n - 1] + leak_rate * self.state_nonlin
        
        return self.states

