'''
Created on Aug 20, 2009

@author: dvrstrae
'''
from utility_functions import get_spectral_radius
import mdp
import numpy
import bimdp

class ReservoirNode(bimdp.BiNode):
    """
    A standard (ESN) reservoir node. Parameters are:
    - input_dim: input dimensionality
    - output_dim: output_dimensionality, i.e. reservoir size
    - spectral_radius: Spectral radius of the internal weight matrix, default: 0.9
    - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
    - bias_scaling: scaling of the bias, a constant input to each neuron, default: 1
    - input_scaling: scaling of the input weight matrix, default: 1
    
    """
    
    def __init__(self, input_dim=1, output_dim=None, spectral_radius=0.9, 
                 nonlin_func = 'tanh', bias_scaling = 0, input_scaling=1, dtype='float64', _instance = 0, node_id=None):
        """ Initializes and constructs a random reservoir.
                
        output_dim -- the number of outputs, which is also the number of
                          neurons in the reservoir
        prototype -- a prototype reservoir which will be cloned with all
                     its parameters
        """
        super(ReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        
        # Set all object attributes
        # Scaling for input weight matrix
        self.input_scaling = input_scaling
        # Scaling for bias weight matrix
        self.bias_scaling = bias_scaling
        # Spectral radius scaling
        self.spectral_radius = spectral_radius
        # Instance ID, used for making different reservoir instantiations with the same parameters
        # Can be ranged over to simulate different 'runs'
        self._instance = _instance
        # Non-linear function
        self.nonlin_func=nonlin_func

        # Call the initialize function to create the weight matrices
        self.initialize()
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def initialize(self):
        """ Initialize the weight matrices of the reservoir node. The w, w_in and w_bias matrices will be created according to the input_scaling, bias_scaling and spectral radius of the node.
        """
        self.w_in = self.input_scaling*(numpy.random.randint(0,2, [self.output_dim, self.input_dim])*2-1) #, output_dim))
        self.w_bias = numpy.ones(self.output_dim)*self.bias_scaling
        self.w = mdp.numx_rand.randn(self.output_dim,self.output_dim)
        # scale it to spectral radius
        self.w *= self.spectral_radius/get_spectral_radius(self.w)
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        
        non_linearity = getattr(mdp.numx, self.nonlin_func)
        steps = x.shape[0]
        self.states = numpy.zeros((steps, self._output_dim), dtype=self._dtype)
        self.state = numpy.zeros(self._output_dim)
        self.state_nonlin = numpy.zeros(self._output_dim)

        for n in range(steps):
            self.state = numpy.dot(self.w, self.state)
            self.state += numpy.dot(self.w_in, x[n])
            self.state += self.w_bias
            self.state_nonlin = non_linearity(self.state)
            self.states[n] = self.state_nonlin
        
        return self.states
 
class LeakyReservoirNode(ReservoirNode):
    """ Reservoir node with leaky integrator neurons (a first-order low-pass filter added to the output of a standard neuron). 
    """

    def __init__(self, input_dim=None, output_dim=None, spec_radius=0.9, 
                nonlin_func = 'tanh', bias_scaling = 0.0, input_scaling=1.0, leak_rate=1.0, dtype='float64'):
        """ Initializes and constructs a random reservoir with leaky-integrator neurons.
               
           output_dim -- the number of outputs, which is also the number of
                             neurons in the reservoir
           prototype -- a prototype reservoir which will be cloned with all
                        its parameters
        """
        super(LeakyReservoirNode, self).__init__(input_dim, output_dim, spec_radius,
                                           nonlin_func, bias_scaling, input_scaling, dtype)
       
        self.leak_rate = leak_rate

    def _execute(self, x):
        ''' Executes simulation with input vector x
        '''
    
        nonlinearity = getattr(mdp.numx, self.nonlin_func)
        steps = x.shape[0]
        self.states = numpy.zeros((steps, self._output_dim), dtype=self._dtype)
        self.state = numpy.zeros(self._output_dim)
        self.state_nonlin = numpy.zeros(self._output_dim)
        leak_rate = self.leak_rate
        
        self.state += numpy.dot(self.w_in, x[0])
        self.state += self.w_bias
        self.state_nonlin = nonlinearity(self.state)
        self.states[0] = leak_rate * self.state_nonlin
        
        for n in range(1, steps):
            self.state = numpy.dot(self.w, self.state)
            self.state += numpy.dot(self.w_in, x[n])
            self.state += self.w_bias
            self.state_nonlin = nonlinearity(self.state)
            
            self.states[n] = (1 - leak_rate) * self.states[n - 1] + leak_rate * self.state_nonlin
        
        return self.states