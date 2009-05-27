import mdp
import aureservoir as au
import numpy
from scipy.lib.lapack import clapack
from scipy.linalg import pinv
from benjamin_nodes import RidgeRegressionNode


class IdentityNode(mdp.Node):
    """ Dummy Node, just to bypass the data
    """
    def is_trainable(self):
        return False


class GaussianIP (object):
    def __init__ (self, parent, mu = 0,
                  sigma = .3, eta = .0001):
	self.a = numpy.ones(parent._output_dim)
	self.initial_W = parent.W
	self.initial_bias = parent.Biasin
	self.b = parent.Biasin 
	self.mu = mu
	self.eta = eta
        self.parent = parent
        self.sigma = sigma

    def execute (self):
	"""
        Gaussian IP
	gaussian_ip (self, parent) 
	x is the current reservoir state before the nonlinearity (activations)
	y is the current reservoir state after the nonlinearity
	ip_a is the a parameter for all the nodes
	ip_b is the b parameter for all the nodes
	eta is the learning rate
	mu is the desired mean of the output distribution
	sigma is the desired standard deviation of the output distribution
	"""
	dddz = 2*self.parent.state_nonlin + (self.parent.state_nonlin-self.mu)* \
              (1+self.parent.state_nonlin*self.parent.state_nonlin)/(self.sigma*self.sigma)
	db = -self.eta*dddz
	da = self.eta/self.a + db * self.parent.state
	self.a += da
	self.b += db
	self.parent.W = self.initial_W * self.a 
	self.parent.Biasin = self.b




class ReservoirSimpleNode(mdp.Node):
    """ Very experimental demo !
    """
    
    def __init__(self, input_dim=None, output_dim=None, spec_radius=0.9, dtype='float64'):
        """ Initializes and constructs a random reservoir.
                
        output_dim -- the number of outputs, which is also the number of
                          neurons in the reservoir
        prototype -- a prototype reservoir which will be cloned with all
                     its parameters
        """
        super(ReservoirSimpleNode, self).__init__(input_dim, output_dim, dtype)
        
        self.Win = numpy.ones((output_dim, input_dim)) #, output_dim))
        self.Biasin = numpy.ones(output_dim)
        self.W = mdp.numx_rand.rand(output_dim,output_dim) * 1.8 - 0.9

        # scale it to spectral radius
        # TODO

        # make a hook object (demo)
        self.hook = GaussianIP(self)
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        steps = x.shape[0]
        states = numpy.zeros((steps, self._output_dim), dtype=self._dtype)
        self.state = numpy.zeros(self._output_dim)
        self.state_nonlin = numpy.zeros(self._output_dim)

        for n in range(steps):
            self.state = numpy.dot(self.W, self.state)
            self.state += numpy.dot(self.Win, x[n])
            self.state += self.Biasin
            self.state_nonlin = numpy.tanh(self.state)

            # call the hook
            self.hook.execute()
            
            states[n] = self.state_nonlin
        
        return states


if __name__ == "__main__":

    inputs = 1
    timesteps = 10

    # construct individual nodes
    reservoir = ReservoirSimpleNode(inputs,20)
    readout = RidgeRegressionNode(1,21,1)

    # build network with MDP framework
    res = mdp.hinet.SameInputLayer([reservoir,
                                    IdentityNode(inputs,inputs)])
    flow = mdp.Flow([res, readout])
    mdp_net = mdp.hinet.FlowNode(flow)

    # make the data
    indata = mdp.numx_rand.rand(timesteps,1) * 1.8 - 0.9
    outdata = numpy.zeros((timesteps,1))
    outdata[1::] = indata[0:-1]

    # train and test it
    mdp_net.train(indata,outdata)
    mdp_net.stop_training()
    testout = mdp_net(indata)

    print indata
    print testout
    print "finished"

