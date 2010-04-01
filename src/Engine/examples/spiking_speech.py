import Engine
import mdp
import pylab
import scipy as sp
from numpy import *
from pyNN.pcsim import *
from pypcsim import *
from NeuroTools import stgen

class PyNNSpeechReservoirNode(mdp.Node):
    """
    An example PyNN reservoir node.
    """    
    def __init__(self, input_dim=1, output_dim=None, dtype=float):
        """
          Create the spiking neural network in PyNN
        """        
        super(PyNNSpeechReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
    
        # === Define parameters ========================================================
    
        self.n = output_dim    # Number of cells
        n_exc = int(0.8 * self.n)
        n_inh = self.n - n_exc
        n_inp = input_dim        
        exc_w = 0.1   # synaptic weight (nA)
        inh_w = -0.5   # synaptic weight (nA)
        input_w = 0.2 # synaptic weight (nA)
        cell_params = {
            'tau_m'      : 20.0, # (ms)
            'tau_syn_E'  : 2.0, # (ms)
            'tau_syn_I'  : 4.0, # (ms)
            'tau_refrac' : 2.0, # (ms)
            'v_rest'     : 0.0, # (mV)
            'v_init'     : 0.0, # (mV)
            'v_reset'    : 0.0, # (mV)
            'v_thresh'   : 20.0, # (mV)
            'cm'         : 0.5}  # (nF)
        dt = 1           # (ms)
        syn_delay = 1.0         # (ms)
    
    
        Cprob_exc = 0.2
        Cprob_inh = 0.2
        Cprob_inp = 0.2
    
        print "Creating the network"
    
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
    
        self.stgen = stgen.StGen(self.rng)
    
    
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
        n_inputs = x.shape[1]
    
        Tstep = 10.0  # in milliseconds
        RateScale = 1e6
    
        in_spikes = []
        for i in range(n_inputs):
            in_spikes.append(self.stgen.inh_poisson_generator(RateScale * x.T[i],
                                                          arange(0, Tstep * len(x.T[i]), Tstep), t_stop=Tstep * len(x.T[i]), array=True)) 
    
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
    
        # calculate the states
        LiqStateTau = 30 # in milliseconds
    
        simDT = 1.0            
    
        exp_kernel = exp(-arange(0, 10 * LiqStateTau, simDT) / LiqStateTau)
        spikes_bin = zeros((len(spikes), int(steps * Tstep / simDT)))
        for i in range(len(spikes)):
            for j in range(len(spikes[i])):                                  
                spikes_bin[i, min(int(spikes[i][j] / simDT), spikes_bin.shape[1] - 1)] += 1.0
        liq_states = array([ convolve(spikes_bin[i], exp_kernel, mode='same') for i in range(len(spikes_bin)) ], dtype=float)
        self.states = swapaxes(liq_states, 0, 1)[::int(Tstep / simDT), :]
    
        return self.states
    
    def _post_update_hook(self, timestep):
        pass


if __name__ == "__main__":

    n_subplots_x, n_subplots_y = 2, 1
    train_frac = .9
    
    [inputs, outputs] = Engine.datasets.analog_speech(indir="data/Lyon128")
    
    n_samples = len(inputs)
    n_train_samples = int(round(n_samples * train_frac))
    n_test_samples = int(round(n_samples * (1 - train_frac)))
    
    input_dim = inputs[0].shape[1]
    
    # construct individual nodes
    reservoir = PyNNSpeechReservoirNode(input_dim, output_dim=400)
    readout = Engine.nodes.RidgeRegressionNode(0.001)
    mnnode = Engine.nodes.MeanAcrossTimeNode()
    
    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout, mnnode])
    
    pylab.subplot(n_subplots_x, n_subplots_y, 1)
    pylab.imshow(inputs[0].T, aspect='auto', interpolation='nearest')
    pylab.title("Cochleogram (input to reservoir)")
    pylab.ylabel("Channel")
    
    
    print "Training..."
    # train and test it
    flow.train([[inputs[0:n_train_samples - 1]], \
                zip(inputs[0:n_train_samples - 1], \
                    outputs[0:n_train_samples - 1]), \
                [None]])
    
    ytrain, ytest = [], []
    print "Applying to trainingset..."
    for xtrain in inputs[0:n_train_samples - 1]:
        ytrain.append(flow(xtrain))
    print "Applying to testset..."
    for xtest in inputs[n_train_samples:]:
        ytest.append(flow(xtest))
    
    pylab.subplot(n_subplots_x, n_subplots_y, 2)
    pylab.plot(reservoir.states)
    pylab.title("Sample reservoir states")
    pylab.xlabel("Timestep")
    pylab.ylabel("Activation")
    
    ymean = sp.array([sp.argmax(sample) for sample in 
                      outputs[n_train_samples:]])
    ytestmean = sp.array([sp.argmax(sample) for sample in ytest])
    
    print "Error: " + str(mdp.numx.mean(Engine.utils.loss_01(ymean,
                                                               ytestmean)))
    pylab.show()
