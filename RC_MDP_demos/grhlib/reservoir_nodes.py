#===============================================================================
# Reservoir Node for the MDP package
#===============================================================================

import mdp
import aureservoir as au
import numpy
from scipy.lib.lapack import clapack
from scipy.linalg import pinv

# TODO:
# - delay und sum algorithmus implementiern (code von testesn.py verwenden)


class ReservoirNode(mdp.Node):
    """ A random initialized recurrent neural network without adaptation.
    """
    
    def __init__(self, input_dim=None, output_dim=None, dtype='float64',
                 prototype=None):
        """ Initializes and constructs a random reservoir.
                
        output_dim -- the number of outputs, which is also the number of
                          neurons in the reservoir
        prototype -- a prototype reservoir which will be cloned with all
                     its parameters
        """
        super(ReservoirNode, self).__init__(input_dim, output_dim, dtype)
        
        if prototype:
            # copy network
            if dtype == 'float64':
                self.reservoir = au.DoubleESN( prototype )
            else:
                self.reservoir = au.SingleESN( prototype )
        else:
            # create new network
            if dtype == 'float64':
                self.reservoir = au.DoubleESN()
            else:
                self.reservoir = au.SingleESN()
        
        # reset input and output dimension
        if input_dim:
            self.reservoir.setInputs(input_dim)
        if output_dim:
            self.reservoir.setSize(output_dim)
        
        # finally initialize the reservoir
        self.reservoir.init()
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        # TODO: make inverse !
        return False
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        steps = x.shape[0]
        states = numpy.zeros((steps, self._output_dim), dtype=self._dtype)
        self.reservoir.collectStates(x.T.copy(),states,0)
        
        return states


class LinearReadoutNode(mdp.Node):
    """ A linear readout for a ReservoirNode.
    
    This node just computes a simple linear regression.
    """
    
    def __init__(self, input_dim=None, output_dim=None,
                 dtype='float64', ignore=0, ignore_ind=0, use_pi=0):
        """ Init the linear readout.

        ignore     --   ignores the first ignore-nr of samples (washout)
                        of the whole training data
        ignore_ind --   ignores the first ignore_ind-nr of samples (washout)
                        of each individual call of train()
        use_pi     --   use pseudo inverse for weight calculation (slower)
        """
        super(LinearReadoutNode, self).__init__(input_dim, output_dim, dtype)
        
        # variables for training
        self.X = numpy.zeros((0,self._input_dim), dtype=self._dtype)
        self.Y = numpy.zeros((0,self._output_dim), dtype=self._dtype)
        self.ignore = ignore
        self.ignore_ind = ignore_ind
        self.use_pi = use_pi
    
    def is_invertible(self):
        return False
    
    def _train(self, x, y):
        """ Collects the training data for learning.
        
        x        --   the input data
        y        --   the corresponding output data (target)
        """
        # append new data to training data
        self.X = numpy.r_[ self.X, x[self.ignore_ind:,:] ]
        self.Y= numpy.r_[ self.Y, y[self.ignore_ind:,:] ]
    
    def _stop_training(self):
        """ Trains the regression weights.
        """
        # calculate regression weights
        # with pseudo inverse (slow): 
        if self.use_pi:
            self.W = numpy.dot(pinv(self.X[self.ignore:,:]), self.Y[self.ignore:,:])
        # or with least square (much faster):
        else:
            if self._dtype == 'float64':
                v,w,s,rank,info = clapack.dgelss(self.X[self.ignore:,:],
                                                 self.Y[self.ignore:,:])
            else:
                v,w,s,rank,info = clapack.sgelss(self.X[self.ignore:,:],
                                                 self.Y[self.ignore:,:])
            self.W = w[0:self._input_dim,:]
        
        # reset data for training
        self.X = numpy.zeros((0,self._input_dim), dtype=self._dtype)
        self.Y = numpy.zeros((0,self._output_dim), dtype=self._dtype)

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        # calculate the regression
        y = numpy.dot(x, self.W)
        return y


class IdentityNode(mdp.Node):
    """ Dummy Node, just to bypass the data
    """
    def is_trainable(self):
        return False


class StateCompressionNode(mdp.Node):
    """ Compresses a variable length signal (e.g. the state sequence of a reservoir)
    into a fixed number of support points.
    
    As used in "Optimization and Applications of Echo State Networks with
    Leaky Integrator Neurons" by Herbert Jaeger
    """
    
    def __init__(self, input_dim=None, output_dim=None, dtype='float64',
                 support_points=5):
        """ Initializes and constructs a random reservoir.
                
        support_points  --  States are segmented into this number of support
                            points. Between discrete states the values are
                            interpolated linearly.
                            NOTE: the output of one input time series is just one
                            vector, so the time structure is lost ! 
        """
        outputs = input_dim * support_points
        
        super(StateCompressionNode, self).__init__(input_dim, outputs, dtype)
        
        self.supp_points = support_points
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with time series vector x.
        This time series will be compressed into one single vector.
        """
        steps = x.shape[0]
        state_size = x.shape[1]
        
        states_compr = numpy.zeros((1,state_size*self.supp_points),
                                   dtype=self._dtype)
        
        # extract only supp_points states per example
        for s in range(self.supp_points-1): # the first states
            ind = (s+1)*steps/float(self.supp_points) - 1
            lower_ind = numpy.floor( ind )
            
            # make an interpolation of states between indices
            a = ind - lower_ind
            state = (1-a) * x[lower_ind,:]
            state += a * x[lower_ind+1,:]
            
            # store values in compressed state matrix            
            ind2 = s * state_size
            states_compr[0,ind2:ind2+state_size] = state
        
        # finally add the last state
        state = x[-1,:]
        ind2 = (self.supp_points-1) * state_size
        states_compr[0,ind2:ind2+state_size] = state

        return states_compr


class VoteAverageNode(mdp.Node):
    """ Averages the inputs for each individual output.
    Can be used to combine individual classifiers by calculating the mean value.
    
    As used in "Optimization and Applications of Echo State Networks with
    Leaky Integrator Neurons" by Herbert Jaeger
    """
    
    def __init__(self, input_dim=None, output_dim=None, dtype='float64'):
        """ Initializes and constructs a random reservoir.
                
        input_dim   --   nr. of features (f1,f2,...) * individual classifiers
                         coded: (f1,f2,f3,...,f1,f2,f3,...)
        output_dim  --   nr. of features (averaged over all classifiers)
        """
        super(VoteAverageNode, self).__init__(input_dim, output_dim, dtype)
        
        # calculate number of individual voters
        if input_dim % output_dim != 0:
            raise mdp.NodeException("input_dim must be a multiple of output_dim !")
        self.voters = input_dim / output_dim
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with time series vector x.
        """
        steps = x.shape[0]
        y = numpy.zeros((steps,self._output_dim), dtype=self._dtype)
        
        for n in range(self._output_dim):
            y[:,n] = x[:,n::self._output_dim].mean(1)
        
        return y

class ReservoirArrayStateComprNode(mdp.Node):
    """ An array of reservoirs with state compression.
    
    This is an array of reservoirs with state compression.
    As used in "Optimization and Applications of Echo State Networks with
    Leaky Integrator Neurons" by Herbert Jaeger
    """
    
    def __init__(self, input_dim=None, output_dim=None, dtype='float64',
                 nr_experts=10, support_points=5, prototype=None):
        """ Init the linear readout.

        nr_experts     --   how many individual reservoirs are in the array
        support_points --   States are segmented into this number of support
                            points. Between discrete states the values are
                            interpolated linearly.
        prototype      --   a prototype reservoir which will be cloned with all
                            its parameters
        """
        super(ReservoirArrayStateComprNode, self).__init__(input_dim, output_dim, dtype)
        
        self.nr_experts = nr_experts
        self.support_points = support_points
        
        res_size = prototype.getSize()       
        expert_array = []
        for n in range(self.nr_experts):
            # make a deep copy and new initialization of all reservoirs
            
            reservoir = ReservoirNode(input_dim, res_size, dtype, prototype)
            compr = StateCompressionNode(res_size + input_dim,
                                         support_points=self.support_points)
            readout = LinearReadoutNode(self.support_points*(res_size+input_dim),
                                        output_dim, ignore=0, use_pi=1)
            
            res = mdp.hinet.SameInputLayer([reservoir,
                                            IdentityNode(input_dim, input_dim)])
            flow = mdp.Flow([res, compr, readout])
            expert_array.append( mdp.hinet.FlowNode(flow) )
        
        # build multiple networks
        experts = mdp.hinet.SameInputLayer( expert_array )
        output_layer = VoteAverageNode(output_dim*self.nr_experts, output_dim)
        self.network = mdp.hinet.FlowNode( mdp.Flow([experts, output_layer]) )
        
    def is_invertible(self):
        return False
    
    def _train(self, x, y):
        self.network.train(x,y)
    
    def _stop_training(self):
        self.network.stop_training()

    def _execute(self, x):
        return self.network(x)
    
    def setNoise(self, noiselevel=0.):
        """ Sets the noise level of all reservoirs.
        """
        for n in range(self.nr_experts):
            self.network._flow[0].nodes[n]._flow[0].nodes[0].reservoir.setNoise(noiselevel)
    
    def resetState(self):
        """ Resets the state of all reservoirs.
        """
        for n in range(self.nr_experts):
            self.network._flow[0].nodes[n]._flow[0].nodes[0].reservoir.resetState()


class SquareStatesNode(mdp.Node):
    """ Adds additional squared states.
    """
    
    def __init__(self, input_dim=None, dtype='float64'):
        """ Adds squared inputs, so the outputs will be two times as long
        as the inputs.
        """
        super(SquareStatesNode, self).__init__(input_dim, input_dim*2, dtype)
            
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        # TODO: make inverse !
        return False
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        y = numpy.c_[x,x**2]
        return y
