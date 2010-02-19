from Engine import datasets 
from Engine import error_measures
from Engine import ode_nodes

import mdp
import pylab

from Engine.linear_nodes import RidgeRegressionNode

import odeproblempython

class OdeLeakyIntegrator(odeproblempython.OdeProblemPython):
	# Initialize the class. This also calls all C++ - constructors.
	def __init__(self, leak_rate = 1.0, **kwargs):
		super(OdeLeakyIntegrator, self).__init__(**kwargs)
        	self.leak_rate = leak_rate
        	
	# function calc_dydt_python. This defines your dydt.
	#  Input: time, y, self.src (self.src is created in ode_nodes.py)
	#  Output: dydt. 
	#  nvar can be used to simulate less variables (nvar <= len(y)).
        def calc_dydt_python( self, t, y, dydt, nvar ):
		dydt[:] = -self.leak_rate*y[:]+self.src[:]
	
	# No need for an update-function. Can be used to change different internal values, delays, ... 
	# More elaborate example following soon, or check RingNetwork.cpp for an example (ask Martin)
        def update_python( self, y ):
        	pass


if __name__ == "__main__":

    inputs = 1
    timesteps = 10000
    system_order = 30;
    washout=30;

    nx=4
    ny=1

    [x,y] = datasets.narma30(sample_len=9000)

    # construct individual nodes
    op = OdeLeakyIntegrator( leak_rate = 0.1 )
    op.set_accuracy(1e-8)
    # op.set_save_every(0.01) #not necessary to save internal values.
    reservoir = ode_nodes.ReservoirNode(inputs,100,odeproblem=op, dt = 0.1, init_dt = 1e-5, dt_min = 1e-6, dt_max = 0.05)
    readout = RidgeRegressionNode(0)

    # build network with MDP framework
    #   flow = mdp.Flow([reservoir, readout], verbose=1)
    flow = mdp.Flow([reservoir], verbose=1)
    RC = mdp.hinet.FlowNode(flow)
    
    #plot the input
    pylab.subplot(nx,ny,1)
    pylab.plot(x[0])
    
    #train the flow 
    #   RC.train(x[0], y[0])
    
    #apply the trained flow to the training data and test data
    #   trainout = RC(x[0])
    #   testout = RC(x[1])
    oderesult = RC(x[0])
    pylab.subplot(nx,ny,2)
    pylab.plot(oderesult[0][:])

    return
    #plot everything
    pylab.subplot(nx,ny,2)
    pylab.plot(trainout, 'r')
    pylab.plot(y[0], 'b')

    pylab.subplot(nx,ny,3)
    pylab.plot(testout[washout:], 'r')
    pylab.plot(y[1][washout:], 'b')
    
    pylab.subplot(nx,ny,4)
    pylab.plot(reservoir.states)
    pylab.show()
    
    print error_measures.nrmse(y[1],testout, discard=washout)
    print "finished"

