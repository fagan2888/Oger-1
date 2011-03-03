"""
This subpackage contains several utility functions. It also contains several error measures and container objects for some commonly used activation functions.
"""

from utility_functions import (get_spectral_radius, empty_n_d_list, butter_coefficients, mfreqz, ConcatenatingIterator, LinearFunction, TanhFunction, LogisticFunction, SoftmaxFunction, SignFunction)
from error_measures import (timeslice, nrmse, nmse, rmse, mse, loss_01, cosine, ce, mem_capacity, threshold_before_error, ber, f_score, _conf_table, _ber, _f_beta)
from mixin import (mix_in, optimize_parameters, select_inputs, enable_washout, disable_washout)
from spiking_utilities import (poisson_gen, spikes_to_states, inputs_to_spikes, exp_kernel)
from confusion_matrix import (ConfusionMatrix, BinaryConfusionMatrix, plot_conf)

# clean up namespace
del utility_functions
del error_measures
del mixin
del spiking_utilities
del confusion_matrix

__all__ = ['get_spectral_radius', 'empty_n_d_list', 'butter_coefficients', 'ConcatenatingIterator', 'LinearFunction', 'TanhFunction', 'LogisticFunction', 'SoftmaxFunction', 'SignFunction',
           'nrmse', 'nmse', 'rmse', 'mse', 'loss_01', 'cosine', 'ce', 'mem_capacity', 'threshold_before_error', 'ber', 'f_score', '_conf_table', '_ber', '_f_beta',
           'mix_in', 'optimize_parameters', 'select_inputs', 'enable_washout',
           'poisson_gen', 'spikes_to_states', 'inputs_to_spikes', 'exp_kernel',
           'ConfusionMatrix', 'BinaryConfusionMatrix', 'plot_conf']
