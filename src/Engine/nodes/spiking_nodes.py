import mdp
try:
    from pyNN.pcsim import *
    from pypcsim import *
except ImportError:
    pass

class SpikingIFReservoirNode(mdp.Node):
    """
    An example PyNN reservoir node.
    """    
    def __init__(self, input_dim, output_dim, dtype, exc_frac, exc_w, inh_w, input_w, cell_params, syn_delay, Cprob_exc, Cprob_inh, Cprob_inp, kernel, inp2spikes_conversion):
        """ Create the spiking neural network in PyNN
        """        
        super(SpikingIFReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        
        self.kernel = kernel
        self.inp2spikes_conversion = inp2spikes_conversion
        
        # === Define parameters ========================================================
        
        self.n = output_dim    # Number of cells
        n_exc = int(exc_frac * self.n)
        n_inh = self.n - n_exc
        n_inp = input_dim        
        
        self.simDT = dt = 1           # (ms)
        
        setup(timestep=dt, max_delay=syn_delay)
        
        rngseed = 1240498
        
        self.rng = NumpyRNG(seed=rngseed)
        
        self.exc_cells = Population((n_exc,), IF_curr_exp, cell_params, "Excitatory_Cells")
        self.inh_cells = Population((n_inh,), IF_curr_exp, cell_params, "Inhibitory_Cells")
        
        self.input_cells = Population((n_inp,), SpikeSourceArray, {'spike_times': array([]) }, "input")
        
        exc_connector = FixedProbabilityConnector(Cprob_exc, weights=exc_w, delays=syn_delay)
        inh_connector = FixedProbabilityConnector(Cprob_inh, weights=inh_w, delays=syn_delay)
        input_connector = FixedProbabilityConnector(Cprob_inp, weights=input_w, delays=syn_delay)
        
        
        self.e2e_conn = Projection(self.exc_cells, self.exc_cells, exc_connector, target='excitatory', rng=self.rng)
        self.e2i_conn = Projection(self.exc_cells, self.inh_cells, exc_connector, target='excitatory', rng=self.rng)
        self.i2e_conn = Projection(self.inh_cells, self.exc_cells, inh_connector, target='inhibitory', rng=self.rng)
        self.i2i_conn = Projection(self.inh_cells, self.inh_cells, inh_connector, target='inhibitory', rng=self.rng)
        
        self.inp_exc_conn = Projection(self.input_cells, self.exc_cells, input_connector, rng=self.rng)
        self.inp_inh_conn = Projection(self.input_cells, self.inh_cells, input_connector, rng=self.rng)
        
        self.exc_cells.record()
        self.inh_cells.record()
    
    
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
        
        Tstep = 10.0  # in milliseconds
        
        in_spikes = inputs_to_spikes(x, self.inp2spikes_conversion)
        
        # setup the new inputs and reset
        for cell, val in zip(self.input_cells, in_spikes):
            setattr(cell, "spike_times", val)        
        
        reset()        
        run(steps * Tstep)
        
        # retrieve the spikes
        exc_rec_spikes = self.exc_cells.getSpikes()
        inh_rec_spikes = self.inh_cells.getSpikes()
        
        inh_id_shift = len(self.exc_cells)
        
        spikes = [ [] for i in range(len(self.exc_cells) + len(self.inh_cells)) ]        
        for id, st in exc_rec_spikes:
            spikes[int(id)].append(st)
        for id, st in inh_rec_spikes:            
            spikes[inh_id_shift + int(id)].append(st)
        
        self.states = spikes_to_states(spikes, self.kernel, steps, Tstep, self.simDT)
        
        return self.states
