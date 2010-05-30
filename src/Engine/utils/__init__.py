"""
This subpackage contains several utility functions. It also contains several error measures and container objects for some commonly used activation functions.
"""

from utility_functions import (get_spectral_radius, empty_n_d_list, ConcatenatingIterator, LinearFunction, TanhFunction, LogisticFunction, SoftmaxFunction, SignFunction)
from error_measures import (timeslice, nrmse, nmse, rmse, mse, loss_01, cosine, ce, mem_capacity)
from mixin import (mix_in, enable_washout)
from spiking_utilities import (poisson_gen, spikes_to_states, inputs_to_spikes, exp_kernel)

# clean up namespace
del utility_functions
del error_measures
del spiking_utilities
__all__ = ['get_spectral_radius', 'empty_n_d_list', 'ConcatenatingIterator','LinearFunction', 'TanhFunction', 'LogisticFunction', 'SoftmaxFunction', 'SignFunction', 'nrmse', 'nmse', 'rmse', 'mse', 'loss_01', 'cosine', 'ce', 'mem_capacity', 'poisson_gen', 'spikes_to_states', 'inputs_to_spikes', 'exp_kernel']
