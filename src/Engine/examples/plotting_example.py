'''
Created on Feb 10, 2010

@author: dvrstrae
'''
import mdp
import Engine

if __name__ == '__main__':
    ''' Example of plotting results of a parameter sweep
        Runs the NARMA 30 task for spectral radius = 0 to 1.5 and plots the error
        The cross-validation error is stored in the variable errors
    '''
    input_size = 1
    [inputs, outputs] = Engine.datasets.narma30()
    data = [inputs, zip(inputs, outputs)]

    # construct individual nodes
    reservoir = Engine.nodes.ReservoirNode(input_size, input_scaling=0.1, output_dim=100)
    readout = Engine.nodes.RidgeRegressionNode(0.001)

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout])

    # 1D plotting example
    print "First a scan of the spectral radius"
    # Nested dictionary
    gridsearch_parameters = {reservoir:{'spectral_radius':mdp.numx.arange(0.6, 1.3, 0.1)}}
    print "gridsearch_parameters = " + str(gridsearch_parameters)
    opt1D = Engine.evaluation.Optimizer(gridsearch_parameters, Engine.utils.nrmse)
    
    # Run the gridsearch
    opt1D.grid_search(data, flow, n_folds=3, cross_validate_function=Engine.evaluation.n_fold_random)
    opt1D.plot_results()

    # 1D plotting example
    print "Then we range over both spectral radius and input scaling"
    gridsearch_parameters = {reservoir:{'spectral_radius':mdp.numx.arange(0.6, 1.3, 0.2), 'input_scaling': mdp.numx.arange(0.5, .8, 0.1)}}
    print "gridsearch_parameters = " + str(gridsearch_parameters)
    opt2D = Engine.evaluation.Optimizer(gridsearch_parameters, Engine.utils.nrmse)

    # Run the gridsearch
    errors = opt2D.grid_search(data, flow, n_folds=3, cross_validate_function=Engine.evaluation.n_fold_random)
    opt2D.plot_results()
