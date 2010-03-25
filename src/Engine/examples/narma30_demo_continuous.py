import mdp
import Engine
import pylab
import Engine.odesolver
import numpy

class OdeLeakyIntegrator(Engine.odesolver.OdeProblemPython):
        # Initialize the class. This also calls all C++ - constructors.
        def __init__(self, leak_rate = 1.0, input_dim = 1, output_dim = 100, **kwargs):
                super(OdeLeakyIntegrator, self).__init__(**kwargs)
                self.leak_rate = leak_rate
                self.src = 1.0
                self.spectral_radius = 0.9
                self.conn_mat = mdp.numx_rand.randn(output_dim,output_dim)
                self.conn_mat *= self.spectral_radius/Engine.utils.get_spectral_radius(self.conn_mat)
                self.conn_mat -= leak_rate*mdp.numx.eye(output_dim,output_dim)
                self.w_in = 1.0*(numpy.random.randint(0,2, [output_dim, input_dim])*2-1)

        # function calc_dydt_python. This defines your dydt.
        #  Input: time, y, self.src (self.src is created in ode_nodes.py)
        #  Output: dydt. 
        #  nvar can be used to simulate less variables (nvar <= len(y)).
        #  Important note: use dydt[:] i.o. dydt.
        #  Might generate swig::directormethodexception if some error occurs here.
        def calc_dydt_python( self, t, y, dydt, nvar ):
                dydt[:]=numpy.dot(self.conn_mat,y[:])+numpy.dot(self.w_in,self.src[:])

        # No need for an update-function. Can be used to change different internal values, delays, ... 
        # More elaborate example following soon, or check RingNetwork.cpp for an example (ask Martin)
        def update_python( self, y ):
                pass


def get_vals(op, index):
        xvals = []
        yvals = []
        y=numpy.zeros(100,numpy.complex128)
        for x in xrange(op.get_internal_steps()+1):
                t=op.get_yvals_and_time_at(y,x)
                xvals.append(t)
                yvals.append(y[index])

        return xvals, yvals

def plot_vals(op, index):
        x, y = get_vals(op, index)
        pylab.plot(x,numpy.real(y))
        pylab.show()

if __name__ == "__main__":

        inputs = 1
        timesteps = 10000
        system_order = 30
        washout=30
        nr_nodes = 50

        nx=4
        ny=1

        [x,y] = Engine.datasets.narma30(sample_len=9000)

        # construct individual nodes
        op = OdeLeakyIntegrator( leak_rate = 1.5, output_dim = nr_nodes )
        op.set_accuracy(1e-8)
        # op.set_save_every(0.01) #not necessary to save internal values.
        reservoir = Engine.nodes.OdeNode(inputs,nr_nodes,odeproblem=op, dt = 0.1, init_dt = 1e-5, dt_min = 1e-6, dt_max = 0.05)
        readout = Engine.nodes.RidgeRegressionNode(0)
        
        #dt = 0.1; init_dt = 1e-5; dt_min = 1e-6; dt_max = 0.05
        #state = numpy.zeros( (100,), complex )
        #op.set_save_every(dt/2.0)
        #op.solve_DE( state, 0,50*dt,init_dt,dt_min,dt_max)
        #plot_vals(op,0)
        
        #exit(0)
        

        # build network with MDP framework
        flow = mdp.Flow([reservoir, readout], verbose=1)
        #  flow = mdp.Flow([reservoir], verbose=1)
        RC = mdp.hinet.FlowNode(flow)

        #plot the input
        pylab.subplot(nx,ny,1)
        pylab.plot(x[0])

        #train the flow 
        RC.train(x[0], y[0])

        #apply the trained flow to the training data and test data
        trainout = RC(x[0])
        testout = RC(x[1])

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

        print Engine.utils.nrmse(y[1],testout, discard=washout)
        print "finished"

