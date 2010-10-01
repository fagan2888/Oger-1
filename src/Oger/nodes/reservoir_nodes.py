import Oger
import mdp
import scipy.signal
import collections
try:
    import cudamat as cm
except:
    pass

# TODO: MDP parallelization assumes that nodes are state-less when executing! The current ReservoirNode
# does not adhere to this and therefor are not parallelizable. A solution is to make the state local. 
# But this again breaks if we start doing some form of on-line learning. In this case we need, or, 
# fork-join but this only work when training (and joining is often not possible), or do not have 
# fork-join in training mode which will force synchronous execution.

# TODO: leaky neuron is also broken when parallel! 

class ReservoirNode(mdp.Node):
    """
    A standard (ESN) reservoir node.
    """
    
    def __init__(self, input_dim=1, output_dim=None, spectral_radius=0.9,
                 nonlin_func=Oger.utils.TanhFunction, bias_scaling=0, input_scaling=1, dtype='float64', _instance=0,
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
        
        self.initial_state = mdp.numx.zeros((1, self.output_dim))        
        
        # Store any externally passed initialization values for w, w_in and w_bias
        self.w_in_initial = w_in
        self.w_initial = w
        self.w_bias_initial = w_bias
        
        # Fields for allocating reservoir weight matrix w, input weight matrix w_in
        # and bias weight matrix w_bias
        self.w_in = []
        self.w = []
        self.w_bias = []
        
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
            self.w_in = self.input_scaling * (mdp.numx.random.rand(self.output_dim, self.input_dim) * 2 - 1)
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
            self.w_bias = self.bias_scaling * (mdp.numx.random.rand(self.output_dim) * 2 - 1)
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
            self.w = mdp.numx.random.randn(self.output_dim, self.output_dim)
            # scale it to spectral radius
            self.w *= self.spectral_radius / Oger.utils.get_spectral_radius(self.w)
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
        
        # Pre-allocate the state vector, adding the initial state
        states = mdp.numx.concatenate((self.initial_state, mdp.numx.zeros((steps, self.output_dim))))
        
        nonlinear_function_pointer = self.nonlin_func.f
        
        # Loop over the input data and compute the reservoir states
        for n in range(steps):
            states[n + 1, :] = nonlinear_function_pointer(mdp.numx.dot(self.w, states[n, :]) + mdp.numx.dot(self.w_in, x[n, :]) + self.w_bias)
            self._post_update_hook(states, x, n)    

        # Return the whole state matrix except the initial state
        return states[1:, :]
    
    def _post_update_hook(self, states, input, timestep):
        """ Hook which gets executed after the state update equation for every timestep. Do not use this to change the state of the 
            reservoir (e.g. to train internal weights) if you want to use parallellization - use the TrainableReservoirNode in that case.
        """
        pass
    
 
class LeakyReservoirNode(ReservoirNode):
    """Reservoir node with leaky integrator neurons (a first-order low-pass filter added to the output of a standard neuron). 
    """

    def __init__(self, leak_rate=1.0, *args, **kwargs):
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
        super(LeakyReservoirNode, self).__init__(*args, **kwargs)
       
        # Leak rate, if 1 it is a standard neuron, lower values give slower dynamics 
        self.leak_rate = leak_rate

    def _post_update_hook(self, states, input, timestep):
        states[timestep + 1, :] = (1 - self.leak_rate) * states[timestep, :] + self.leak_rate * states[timestep + 1, :]

class BandpassReservoirNode(ReservoirNode):
    """Reservoir node with bandpass neurons (an Nth-order band-pass filter added to the output of a standard neuron). 
    """ 
    def __init__(self, b=mdp.numx.array([[1]]), a=mdp.numx.array([[0]]), *args, **kwargs):
        """Initializes and constructs a random reservoir with band-pass neurons.
           Parameters are:
                - input_dim: input dimensionality
                - output_dim: output_dimensionality, i.e. reservoir size
                - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
                - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
                - input_scaling: scaling of the input weight matrix, default: 1
                - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
                - b: array of coefficients for the numerator of the IIR filter
                - a: array of coefficients for the denominator of the IIR filter
            
            Weight matrices are either generated randomly or passed at construction time.
            if w, w_in or w_bias are not given in the constructor, they are created randomly:
                - input matrix : input_scaling * uniform weights in [-1, 1]
                - bias matrix :  bias_scaling * uniform weights in [-1, 1]
                - reservoir matrix: gaussian weights rescaled to the desired spectral radius
            If w, w_in or w_bias were given as a numpy array or a function, these
            will be used as initialization instead.               
        """
        self.a = a
        self.b = b
        super(BandpassReservoirNode, self).__init__(*args, **kwargs)
        self.input_buffer = collections.deque([mdp.numx.zeros(self.output_dim)] * self.b.shape[1], maxlen=self.b.shape[1])
        self.output_buffer = collections.deque([mdp.numx.zeros(self.output_dim)] * self.a.shape[1], maxlen=self.a.shape[1])

    def _post_update_hook(self, states, input, timestep):
        self.input_buffer.appendleft(states[timestep + 1, :])
        t1 = mdp.numx.sum(self.b * mdp.numx.array(self.input_buffer).T, axis=1)
        t2 = mdp.numx.sum(self.a * mdp.numx.array(self.output_buffer).T, axis=1)
        states[timestep + 1, :] = t1 - t2
        self.output_buffer.appendleft(states[timestep, :])
        
        
        
class TrainableReservoirNode(ReservoirNode):
    """A reservoir node that allows on-line training of the internal connections. Use
    this node for this purpose instead of implementing the _post_update_hook in the
    normal ReservoirNode as this is incompatible with parallelization. 
    """
    def is_trainable(self):
        return True
    
    def _train(self, x):
        states = self._execute(x)
        self._post_train_hook(states, input)
        
    def _post_update_hook(self, states, input, timestep):
        super(TrainableReservoirNode, self)._post_update_hook(states, input, timestep)
        if self.is_training():
            self._post_train_update_hook(states, input, timestep) 

    def _post_train_update_hook(self, states, input, timestep):
        """Implement this function for on-line training after each time-step
        """ 
        pass

    def _post_train_hook(self, states, input):
        """Implement this function for training after each time-series
        """ 
        pass

class HebbReservoirNode(TrainableReservoirNode):
    """This node does nothing good, it is just a demo of training a reservoir.
    """
    def _post_train_update_hook(self, states, input, timestep):
        self.w -= 0.01 * mdp.utils.mult(states[timestep + 1:timestep + 2, :].T, states[timestep:timestep + 1, :])
        self.w_in -= 0.01 * mdp.utils.mult(states[timestep + 1:timestep + 2, :].T, input[timestep:timestep + 1, :])
        self.w_bias -= 0.01 * states[timestep + 1, :];

class FeedbackReservoirNode(ReservoirNode):
    """This is a reservoir node that can be used for setups that use output 
    feedback. Note that because state needs to be stored in the Node object,
    this Node type is not parallelizable using threads.
    """
    
    def __init__(self, reset_states=True, **kwargs):
        super(FeedbackReservoirNode, self).__init__(**kwargs)
        self.reset_states = reset_states
        self.states = mdp.numx.zeros((1, self.output_dim))
        
    def _execute(self, x):        
        # Set the initial state of the reservoir
        # if self.reset_states is true, initialize to zero,
        # otherwise initialize to the last time-step of the previous execute call (for freerun)
        if self.reset_states:
            self.initial_state = mdp.numx.zeros((1, self.output_dim))
        else:
            self.initial_state = mdp.numx.atleast_2d(self.states[-1, :])

        self.states = super(FeedbackReservoirNode, self)._execute(x)

        return self.states

def get_specrad(Ac):
        """Get spectral radius of A using the power method."""
    
        m_size = Ac.shape[0]
    
        x = np.random.normal(0, 1, (m_size, 1))
    
        x = x / np.linalg.norm(x)
        x = cm.CUDAMatrix(x)
    
        y = cm.empty((m_size, 1))
        diff = 200
        eps = 1e-3
        b = 1e10
        c = 1e9
        r = 1e9
        max_its = 1e6
    
        n_its = 0
    
        while diff > eps and n_its < max_its:
            cm.dot(Ac, x, target=y)
            norm = y.euclid_norm()
            y.divide(norm, target=x)
            a = cm.dot(y.T, x).asarray()
            c = cm.dot(x.T, x).asarray()
            diff = np.abs(a - b)
            b = float(a)
            n_its += 1
    
        specrad = float(a / c)
        print 'Spectral radius:', specrad, 'Number of iterations:', n_its
        return float(a / c)


class GPUReservoirNode(mdp.Node):
    def __init__(self, input_dim, output_dim, spectral_radius=.9, leak_rate=1,
                                             input_scaling=1, bias_scaling=0):
        super(GPUReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim,)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leak_rate = leak_rate
        w = mdp.numx.random.normal(0, 1, (output_dim, output_dim))
        w_in = mdp.numx.random.uniform(-1, 1, (output_dim, input_dim)) * input_scaling
        if output_dim < 1500:
            l = mdp.numx.linalg.eigvals(w)
            r = mdp.numx.amax(mdp.numx.absolute(l))
            w = w * (spectral_radius / r)
            self.w = cm.CUDAMatrix(w)
        else:
            self.w = cm.CUDAMatrix(w)
            r = get_specrad(self.w)
            self.w.mult(spectral_radius / r)
        bias = mdp.numx.random.normal(0, 1, (output_dim, 1)) * bias_scaling
        self.w_in = cm.CUDAMatrix(w_in)
        self.bias = cm.CUDAMatrix(bias)
        self.current_state = cm.empty((self.output_dim, 1))
        self.new_state = cm.empty((self.output_dim, 1))


    def _execute(self, x):
        n = x.shape[0]
        # Do everything in transpose because row indexing is very expensive.
        x_T = x.transpose()
        self.states = cm.empty((self.output_dim, n + 1))

        self.states.set_col_slice(0, 1, cm.CUDAMatrix(mdp.numx.zeros((self.output_dim, 1))))
        temp = cm.empty((self.output_dim, 1))
        bias = self.bias

        for i in range(n):

            self.current_state = self.states.get_col_slice(i, i + 1)
            self.new_state = self.states.get_col_slice(i + 1, i + 2)

            cm.dot(self.w, self.current_state, self.new_state)
            self.new_state.add_dot(self.w_in, x_T.get_col_slice(i, i + 1))

            # I should either make this trainable or leave it out.
            #self.new_state.add(bias)

            self.new_state.apply_tanh()

            # Add leakrate
            #self.new_state.mult(self.leak_rate)
            #self.new_state.add_mult(self.current_state, 1 - self.leak_rate)

            # Note that the states are shifted in time and that the first state
            # is zero.
            states_out = self.states.get_col_slice(0, n)

        return states_out.transpose()

    def update(self, input, previous):
        new_state = cm.empty(previous.shape)
        cm.dot(self.w, previous, new_state)
        new_state.add_dot(self.w_in, input)
        #new_state.add(self.bias)
        new_state.apply_tanh()
        return new_state
