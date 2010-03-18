"""
A collection of some trainers that use gradient information.

The convention is to minimize the objective (error) function.
"""

import mdp
from mdp import numx

def gradient_descent(func, params, **kwargs):
    """Perform gradient descent over params using the gradient returned by func.

    Arguments in **kwargs:

        learning_rate -- size of the gradient steps (default = .001)
        
        momentum -- momentum term (default = 0)

        epochs -- number of times to do updates on the same data (default = 1)

        decay -- weight decay term (default = 0)
    """

    learning_rate = kwargs.get('learning_rate', .001)
    momentum = kwargs.get('momentum', 0)
    epochs = kwargs.get('epochs', 1)
    decay = kwargs.get('decay', 0)

    dparams = numx.zeros(params.shape)
    updated_params = params

    for epoch in range(epochs):
        gradient = func(updated_params)[0]
        dparams = momentum * dparams - learning_rate * gradient
        updated_params += dparams - decay * updated_params

    return updated_params


