'''
Created on Aug 20, 2009

@author: dvrstrae
'''
from utility_functions import get_spectral_radius
import mdp
import numpy

class ReservoirNode(mdp.Node):
    """
    A standard (ESN) reservoir node.
    """
    
    def __init__(self, input_dim=1, output_dim=None, spectral_radius=0.9,
                 nonlin_func='tanh', bias_scaling=0, input_scaling=1, dtype='float64', _instance=0,
                 w_in=None, w=None, w_bias=None):
        """ Initializes and constructs a random reservoir.
        Parameters are:
        - input_dim: input dimensionality
        - output_dim: output_dimensionality, i.e. reservoir size
        - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
        - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
        - input_scaling: scaling of the input weight matrix, default: 1
        - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
        
        Weight matrices are either generated randomly or passed at construction time.
        if w, w_in or w_bias are not given in the constructor, they are created randomly:
            - input matrix : input_scaling * uniform weights in [-1, 1]
            - bias matrix :  bias_scaling * uniform weights in [-1, 1]
            - reservoir matrix: gaussian weights rescaled to the desired spectral radius
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
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
        self.w_in_initial = w_in
        self.w_initial = w
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
        """ Initialize the weight matrices of the reservoir node. If no 
        arguments for w, w_in and w_bias matrices were given at construction
        time, they will be created as follows:
        - input matrix : input_scaling * uniform weights in [-1, 1]
        - bias matrix :  bias_scaling * uniform weights in [-1, 1]
        - reservoir matrix: gaussian weights rescaled to the desired spectral radius
        If w, w_in or w_bias were given as a numpy array or a function, these
        will be used as initialization instead.
        """
        # Initialize input weight matrix
        if self.w_in_initial is None:
            # Initialize it to uniform random values using input_scaling
            self.w_in = self.input_scaling * (numpy.random.rand(self.output_dim, self.input_dim) * 2 - 1)
        else:
            if callable(self.w_in_initial):
                self.w_in = self.w_in_initial() # If it is a function, call it
            else:
                self.w_in = self.w_in_initial # else just copy it
        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w_in.shape != (self.output_dim, self.input_dim):
            exception_str = 'Shape of given w_in does not match input/output dimensions of node. '
            exception_str += 'Input dim: ' + str(self.input_dim) + ', output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w_in: ' + str(self.w_in.shape)
            raise mdp.NodeException(exception_str)
                   
        # Initialize bias weight matrix
        if self.w_bias_initial is None:
            # Initialize it to uniform random values using input_scaling
            self.w_bias = self.bias_scaling * (numpy.random.rand(self.output_dim) * 2 - 1)
        else:    
            if callable(self.w_bias_initial):
                self.w_bias = self.w_bias_initial() # If it is a function, call it
            else:
                self.w_bias = self.w_bias_initial   # else just copy it

        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w_bias.shape != (self.output_dim,):
            exception_str = 'Shape of given w_bias does not match input/output dimensions of node. '
            exception_str += 'Input dim: ' + str(self.input_dim) + ', output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w_bias: ' + str(self.w_bias.shape)
            raise mdp.NodeException(exception_str)
            
        # Initialize reservoir weight matrix
        if self.w_initial is None:
            self.w = numpy.random.randn(self.output_dim, self.output_dim)
            # scale it to spectral radius
            self.w *= self.spectral_radius / get_spectral_radius(self.w)
        else:
            if callable(self.w_initial):
                self.w = self.w_initial() # If it is a function, call it
            else:
                self.w = self.w_initial   # else just copy it
        
        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w.shape != (self.output_dim, self.output_dim):
            exception_str = 'Shape of given w does not match input/output dimensions of node. '
            exception_str += 'Output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w: ' + str(self.w_in.shape)
            raise mdp.NodeException(exception_str)
            
    
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
            self._post_update_hook(n)    

        # Return the whole state matrix except the initial state
        return self.states[1:, :]
    
    def _post_update_hook(self, timestep):
        pass
 
class LeakyReservoirNode(ReservoirNode):
    """Reservoir node with leaky integrator neurons (a first-order low-pass filter added to the output of a standard neuron). 
    """

    def __init__(self, input_dim=1, output_dim=None, spec_radius=0.9,
                nonlin_func='tanh', bias_scaling=0.0, input_scaling=1.0, leak_rate=1.0, dtype='float64'):
        """Initializes and constructs a random reservoir with leaky-integrator neurons.
           Parameters are:
            - input_dim: input dimensionality
            - output_dim: output_dimensionality, i.e. reservoir size
            - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
            - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
            - input_scaling: scaling of the input weight matrix, default: 1
            - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
            - leak_rate: if 1 it is a standard neuron, lower values give slower dynamics
            
            Weight matrices are either generated randomly or passed at construction time.
            if w, w_in or w_bias are not given in the constructor, they are created randomly:
                - input matrix : input_scaling * uniform weights in [-1, 1]
                - bias matrix :  bias_scaling * uniform weights in [-1, 1]
                - reservoir matrix: gaussian weights rescaled to the desired spectral radius
            If w, w_in or w_bias were given as a numpy array or a function, these
            will be used as initialization instead.               
        """
        super(LeakyReservoirNode, self).__init__(input_dim, output_dim, spec_radius,
                                           nonlin_func, bias_scaling, input_scaling, dtype)
       
        #Initial value for lowpass filter
        self.previous_state = numpy.zeros(output_dim)
        # Leak rate, if 1 it is a standard neuron, lower values give slower dynamics 
        self.leak_rate = leak_rate

    def _post_update_hook(self, timestep):
        self.states[timestep + 1, :] = (1 - self.leak_rate) * self.previous_state + self.leak_rate * self.states [timestep + 1, :]
        self.previous_state = self.states[timestep + 1, :]
