'''
Created on Feb 10, 2010

@author: dvrstrae
'''
import mdp
import Engine

if __name__ == '__main__':

    [inputs, outputs] = Engine.datasets.narma30()
    data = [inputs, zip(inputs, outputs)]

    # construct individual nodes
    reservoir = Engine.nodes.ReservoirNode(input_scaling=0.1, output_dim=100)
    readout = mdp.nodes.LinearRegressionNode()

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout])

    # 1D plotting example
    print "First a scan of the spectral radius, instantiating 5 reservoirs for each setting."
    # Nested dictionary
    gridsearch_parameters = {reservoir:{'_instance':range(5), 'spectral_radius':mdp.numx.arange(0.6, 1.3, 0.1)}}
    print "gridsearch_parameters = " + str(gridsearch_parameters)
    opt1D = Engine.evaluation.Optimizer(gridsearch_parameters, Engine.utils.nrmse)
    
    # Run the gridsearch
    opt1D.grid_search(data, flow, n_folds=3, cross_validate_function=Engine.evaluation.n_fold_random)
    
    # Plot the results, after taking the mean over the reservoir instances
    opt1D.plot_results([(reservoir, '_instance')])

    # 1D plotting example
    print "Then we range over both spectral radius and input scaling, instantiating 5 reservoirs for each setting."
    gridsearch_parameters = {reservoir:{'spectral_radius':mdp.numx.arange(0.6, 1.3, 0.2),
                                        'input_scaling': mdp.numx.arange(0.1, 0.5, 0.1),
                                        '_instance':range(5)}}
    print "gridsearch_parameters = " + str(gridsearch_parameters)
    opt2D = Engine.evaluation.Optimizer(gridsearch_parameters, Engine.utils.nrmse)

    # Run the gridsearch
    errors = opt2D.grid_search(data, flow, n_folds=3, cross_validate_function=Engine.evaluation.n_fold_random)
    
    # Plot the results, after taking the mean over the reservoir instances
    opt2D.plot_results([(reservoir, '_instance')])
