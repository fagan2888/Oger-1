"""
	ode_nodes.py

	Ordinary differential equations.
"""

import odeproblempython
import mdp
import numpy



class OdeNode(mdp.Node):
    """
     Very experimental demo !
    
    """
    
    def __init__(self, input_dim=1, output_dim=None, odeproblem = None, dt = None, init_dt = None, dt_min = None, dt_max = None, dtype='float64'):
        """ Initializes and constructs a reservoir in continuous time, using differential equations.
            dt is the external timestep after which we save the states.
	    init_dt is the starting dt for which the ode-problem is solved. Internally, the dt can be different (it's adaptive timestepping).

        output_dim -- the number of outputs, which is also the number of
                          neurons in the reservoir.
        """
        super(OdeNode, self).__init__(input_dim, output_dim, dtype)
        
	self.odeproblem = odeproblem
	self.dt = dt
	self.init_dt = init_dt
	self.dt_min = dt_min
	self.dt_max = dt_max
        self.initialize()
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def initialize(self):
	pass
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        """ Executes simulation with input vector x.
        """
        
        steps = x.shape[0]
        self.states = numpy.zeros((steps, self._output_dim), dtype=self._dtype)
        self.state = numpy.zeros(self._output_dim)

	# Alternative: put set_save_every to dt and simulate from t1 to t2 (=t1+dt*steps), without the for-loop.
	# Then interpolating _yn on dt. This approach, however, means you cannot interrupt your simulation in between timesteps 
	# (if you want to add feedback from outside the reservoir).

	dt_current = self.init_dt
        for n in range(steps):
	    t1 = n*self.dt
	    self.odeproblem.src = x[n]
            self.odeproblem.solve_DE(self.state,t1,t1+dt,dt_current,dt_min,dt_max)

            # Get the altered internal dt (smart to use that one to continue simulating in the next step i.o. init_dt)
	    dt_curent = self.odeproblem.get_dt()
	    self.odeproblem.set_dt_current( dt_current )
            
            self.states[n] = self.state
        
        return self.states
 
