import mdp
try:
    from pyNN.pcsim import *
    from pypcsim import *
except:
    pass

try:
    import brian_no_units
    import brian
except ImportError:
    pass

from datetime import datetime

import scipy

class BrianIFReservoirNode(mdp.Node):
    def __init__(self, input_dim, output_dim, dtype, input_scaling=100, input_conn_frac=.5, dt=1, we_scaling=2, wi_scaling=.5, we_sparseness=.1, wi_sparseness=.1):
        super(BrianIFReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.taum = 20 * brian.ms
        self.taue = 5 * brian.ms
        self.taui = 10 * brian.ms
        self.Vt = 15 * brian.mV
        self.Vr = 0 * brian.mV
        self.frac_e = .75
        self.input_scaling = input_scaling
        self.input_conn_frac = input_conn_frac
        self.dt = dt
        self.we_scaling = we_scaling
        self.wi_scaling = wi_scaling
        self.we_sparseness = we_sparseness
        self.wi_sparseness = wi_sparseness

        self.eqs = brian.Equations('''
              dV/dt  = (I-V+ge-gi)/self.taum : volt
              dge/dt = -ge/self.taue    : volt 
              dgi/dt = -gi/self.taui    : volt
              I: volt
              ''')
        self.G = brian.NeuronGroup(N=output_dim, model=self.eqs, threshold=self.Vt, reset=self.Vr)
        self.Ge = self.G.subgroup(int(scipy.floor(output_dim * self.frac_e))) # Excitatory neurons 
        self.Gi = self.G.subgroup(int(scipy.floor(output_dim * (1 - self.frac_e))))

        self.internal_conn = brian.Connection(self.G, self.G)
        self.we = self.we_scaling * scipy.random.rand(len(self.Ge), len(self.G)) * brian.nS
        self.wi = self.wi_scaling * scipy.random.rand(len(self.Ge), len(self.G)) * brian.nS

        self.Ce = brian.Connection(self.Ge, self.G, 'ge', sparseness=self.we_sparseness, weight=self.we)
        self.Ci = brian.Connection(self.Gi, self.G, 'gi', sparseness=self.wi_sparseness, weight=self.wi)

        #self.internal_conn.connect(self.G, self.G, self.w_res)

        self.Mv = brian.StateMonitor(self.G, 'V', record=True, timestep=10)
        self.Ms = brian.SpikeMonitor(self.G, record=True)
        self.w_in = self.input_scaling * (scipy.random.rand(self.output_dim, self.input_dim)) * (scipy.random.rand(self.output_dim, self.input_dim) < self.input_conn_frac)
        self.network = brian.Network(self.G, self.Ce, self.Ci, self.Ge, self.Gi, self.Mv, self.Ms)

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        #self.network = brian.Network(self.G, self.Ce, self.Ci, self.Ge, self.Gi, self.Mv, self.Ms)
        self.network.reinit()
        brian.clear(True)
        #brian.reinit_default_clock() 
        #G  = self.G
        #Ce = self.Ce
        #Ci = self.Ci
        #Ge = self.Ge
        #Gi = self.Gi
        #Mv = self.Mv
        #Ms = self.Ms
        #brian.set_group_var_by_array(self.G, 'I', 100 * scipy.dot(x, self.w_in.T) * brian.mV, start=0,  dt=self.dt * brian.ms)
        self.G.I = brian.TimedArray(100 * scipy.dot(x, self.w_in.T) * brian.mV, dt=1 * brian.ms)
        time = (x.shape[0] + 1) * self.dt
        print "Running for %d ms." % time
        n = datetime.now()
        self.network.run(time * brian.ms)
        n2 = datetime.now()
        print 'Ran for %f ms/ms.' % (float((n2 - n).microseconds) / time)
        retval = scipy.signal.resample(self.Mv.values.T, x.shape[0])
        return retval

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

