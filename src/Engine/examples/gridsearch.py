'''
Created on Feb 9, 2010

@author: dvrstrae
'''

import mdp
from Engine import reservoir_nodes
from Engine import linear_nodes
from Engine import datasets
from Engine import error_measures
from Engine import optimizer

if __name__ == '__main__':
    ''' Example of doing a grid-search
        Runs the NARMA 30 task for bias values = 0 to 2 with 0.5 stepsize and spectral radius = 0.1 to 1 with stepsize 0.5
    '''
    input_size = 1
    [x,y] = datasets.narma30()

    # construct individual nodes
    reservoir = reservoir_nodes.ReservoirNode(input_size,10)
    readout = linear_nodes.RidgeRegressionNode(0)

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout], verbose=1)
    RC = mdp.hinet.FlowNode(flow)

    # Nested dictionary
    gridsearch_parameters = {reservoir:{'bias': mdp.numx.arange(0, 2, 0.5), 'spec_radius':mdp.numx.arange(0.1, 1, 0.5)}}

    # Instantiate an optimizer
    opt = optimizer.Optimizer(gridsearch_parameters, error_measures.nrmse)
    
    # Do the grid search
    opt.grid_search(x,y, RC)
