import mdp
from numpy import array, zeros, convolve, swapaxes
from NeuroTools import stgen
try:
    from pyNN.pcsim import *
    from pypcsim import *
except ImportError:
    pass

class poisson_gen:
    '''Container class for a poisson generator signal-to-spiketrain convertor
    '''
    def __init__(self, rngseed, RateScale = 1e6, Tstep = 10):
        ''' Create a poisson generator, using the given seed for the random number generator
        '''
        self.RateScale = RateScale
        self.Tstep = Tstep
        self.rng = NumpyRNG(seed=rngseed)
        self.stgen = stgen.StGen(self.rng)

    def __call__(self, xi):
        '''
            PoissonGen(xi) -> spikes
            Return a poisson spike train for the given signal xi
        '''
        return self.stgen.inh_poisson_generator(self.RateScale * xi, mdp.numx.arange(0, self.Tstep * len(xi), self.Tstep), t_stop= self.Tstep * len(xi), array=True)


def spikes_to_states(spikes, kernel, steps, Tstep, simDT):
    '''
        spikes_to_states(spikes, kernel, steps, Tstep, simDT) -> states
        Convert spikes to liquid states using a given convolution kernel
        Input arguments: 
        - spikes: the spike train to be converted
        - kernel: the convolution kernel (a function of a vector which returns a vector)
        - steps: number of timesteps in the resulting vector in the analog domain
        - Tstep: timestep (sampling period) of the resulting vector in the analog domain
        - simDt: simulation timestep
    '''
    spikes_bin = zeros((len(spikes), int(steps * Tstep / simDT)))
    
    for i in range(len(spikes)):
        for j in range(len(spikes[i])):                                  
            spikes_bin[i, min(int(spikes[i][j] / simDT), spikes_bin.shape[1] - 1)] += 1.0
    
    liq_states = array([ convolve(spikes_bin[i], kernel, mode='same') for i in range(len(spikes_bin)) ], dtype=float)
    states = swapaxes(liq_states, 0, 1)[::int(Tstep / simDT), :]
    return states


def inputs_to_spikes(x, inp2spikes_conversion):
    ''' 
        inputs_to_spikes(x, inp2spikes_conversion) -> in_spikes
        Convert an N-d analog signal to spiketrains, using the given spiketrain conversion function
        Input arguments:
        - x: the dataset to be converted (a numpy array)
        - inp2spikes_conversion: the function that performs the spike conversion
    '''
    n_inputs = x.shape[1]
    in_spikes = []
    for i in range(n_inputs):
            in_spikes.append(inp2spikes_conversion(x.T[i]))
    return in_spikes


def exp_kernel(tau, dt):
    '''
        exp_kernel(tau, dt) -> kernel_result
        Exponential kernel for filtering spike trains
    '''
    return mdp.numx.exp(-mdp.numx.arange(0, 10 * tau, dt) / tau)
