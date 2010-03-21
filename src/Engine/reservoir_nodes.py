'''
Created on Aug 20, 2009

@author: dvrstrae
'''
from utility_functions import get_spectral_radius
import mdp
import numpy

class ReservoirNode(mdp.Node):
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
                 nonlin_func='tanh', bias_scaling=0, input_scaling=1, dtype='float64', _instance=0,
                 w_in=None, w=None, w_bias=None):
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
        self.nonlin_func = nonlin_func
        self.reset_states = True
        self.collected_states = []
        
        # Store any externally passed initialization values for w, w_in and w_bias
        if w_in is not None and w_in.shape != (output_dim, input_dim):
            raise mdp.NodeException('Shape of given w_in does not match input/output dimensions of node.')
        else:
            self.w_in_initial = w_in
        if w is not None and w.shape != (output_dim, output_dim):
            raise mdp.NodeException('Shape of given w does not match input/output dimensions of node.')
        else:
            self.w_initial = w
        if w_bias is not None and w_bias.shape != (output_dim,):
            raise mdp.NodeException('Shape of given w_bias does not match input/output dimensions of node.')
        else:
            self.w_bias_initial = w_bias
        
        # Fields for allocating reservoir weight matrix w, input weight matrix w_in
        # and bias weight matrix w_bias
        self.w_bias = []
        self.w_in = []
        self.w = []

        # Call the initialize function to create the weight matrices
        self.initialize()
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def initialize(self):
        """ Initialize the weight matrices of the reservoir node. The w, w_in and w_bias matrices will be created according to the input_scaling, bias_scaling and spectral radius of the node.
        """
        # Initialize input weight matrix
        if self.w_in_initial is None:
            self.w_in = self.input_scaling * (numpy.random.randint(0, 2, [self.output_dim, self.input_dim]) * 2 - 1)
        else:
            self.w_in = self.w_in_initial
            
        # Initialize bias weight matrix
        if self.w_bias_initial is None:     
            self.w_bias = numpy.ones(self.output_dim) * self.bias_scaling
        else:
            self.w_bias = self.w_bias_initial
        
        # Initialize reservoir weight matrix
        if self.w_initial is None:
            self.w = numpy.random.randn(self.output_dim, self.output_dim)
            # scale it to spectral radius
            self.w *= self.spectral_radius / get_spectral_radius(self.w)
        else:
            self.w = self.w_initial
        
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        steps = x.shape[0]
        
        # Set the initial state of the reservoir
        # if self.reset_states is true, initialize to zero,
        # otherwise initialize to the last time-step of the previous execute call (for freerun)
        if self.reset_states:
            initial_state = numpy.zeros((1, self.output_dim))
        else:
            initial_state = numpy.atleast_2d(self.states[-1, :])
        
        # Pre-allocate the state vector, adding the initial state
        self.states = numpy.concatenate((initial_state, numpy.zeros((steps, self.output_dim))))
       
        nonlinear_function_pointer = getattr(mdp.numx, self.nonlin_func)
        # Loop over the input data and compute the reservoir states
        for n in range(steps):
            self.states[n + 1, :] = nonlinear_function_pointer(numpy.dot(self.w, self.states[n, :]) + numpy.dot(self.w_in, x[n, :]) + self.w_bias)    

        # Return the whole state matrix except the initial state
        return self.states[1:, :]
 
class LeakyReservoirNode(ReservoirNode):
    """ Reservoir node with leaky integrator neurons (a first-order low-pass filter added to the output of a standard neuron). 
    """

    def __init__(self, input_dim=1, output_dim=None, spec_radius=0.9,
                nonlin_func='tanh', bias_scaling=0.0, input_scaling=1.0, leak_rate=1.0, dtype='float64'):
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
        steps = x.shape[0]
        
        # Set the initial state of the reservoir
        # if self.reset_states is true, initialize to zero,
        # otherwise initialize to the last time-step of the previous execute call (for freerun)
        if self.reset_states:
            initial_state = numpy.zeros((1, self.output_dim))
        else:
            initial_state = numpy.atleast_2d(self.states[-1, :])
        
        # Pre-allocate the state vector, adding the initial state
        self.states = numpy.concatenate((initial_state, numpy.zeros((steps, self.output_dim))))
       
        nonlinear_function_pointer = getattr(mdp.numx, self.nonlin_func)
        # Loop over the input data and compute the reservoir states
        for n in range(steps):
            # First compute the output of the non-leaky neuron (standard sigmoid)
            unfiltered_output = nonlinear_function_pointer(numpy.dot(self.w, self.states[n, :]) + numpy.dot(self.w_in, x[n, :]) + self.w_bias)
            # Apply the low-pass filter
            self.states[n + 1, :] = (1 - self.leak_rate) * self.states[n, :] + self.leak_rate * unfiltered_output

        # Return the whole state matrix except the initial state
        return self.states[1:, :]    
