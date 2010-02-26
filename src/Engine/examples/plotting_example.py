'''
Created on Feb 10, 2010

@author: dvrstrae
'''
import mdp
from Engine import reservoir_nodes
from Engine import linear_nodes
from Engine import datasets
from Engine import error_measures
from Engine import optimizer

if __name__ == '__main__':
    ''' Example of plotting results of a parameter sweep
        Runs the NARMA 30 task for spectral radius = 0 to 1.5 and plots the error
        The cross-validation error is stored in the variable errors
    '''
    input_size = 1
    [x,y] = datasets.narma30()

    # construct individual nodes
    reservoir = reservoir_nodes.ReservoirNode(input_size, input_scaling=0.1, output_dim=100)
    readout = linear_nodes.RidgeRegressionNode(0.001)

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout], verbose=1)
    RC = mdp.hinet.FlowNode(flow)

    # 1D plotting example
    print "First a scan of the spectral radius : gridsearch_parameters = {reservoir:{'spec_radius':mdp.numx.arange(0.1, 1.2, 0.1)}}"
    # Nested dictionary
    #gridsearch_parameters = {reservoir:{'spec_radius':mdp.numx.arange(0.1, 1.2, 0.1), 'instance': range(10)}}
    gridsearch_parameters = {reservoir:{'spec_radius':mdp.numx.arange(0.6, 1.2, 0.1)}}
    opt1D = optimizer.Optimizer(gridsearch_parameters, error_measures.nrmse)

    
    # Run the gridsearch
    opt1D.grid_search(x,y, RC, n_folds = 3)
    
    opt1D.plot_results()

    # 1D plotting example
    print "Then we range over both spectral radius and input scaling"
    gridsearch_parameters = {reservoir:{'spec_radius':mdp.numx.arange(0.6, 1.2, 0.2), 'input_scaling': mdp.numx.arange(0.5, .7, 0.1)}}
    opt2D = optimizer.Optimizer(gridsearch_parameters, error_measures.nrmse)

    # Run the gridsearch
    errors = opt2D.grid_search(x,y, RC, n_folds = 3)

    opt2D.plot_results()
