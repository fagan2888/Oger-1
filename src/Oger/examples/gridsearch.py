import mdp
import Oger

if __name__ == '__main__':
    ''' Example of doing a grid-search
        Runs the NARMA 30 task for bias values = 0 to 2 with 0.5 stepsize and spectral radius = 0.1 to 1 with stepsize 0.5
    '''
    input_size = 1
    inputs, outputs = Oger.datasets.narma30()

    data = [inputs, zip(inputs, outputs)]

    # construct individual nodes
    reservoir = Oger.nodes.ReservoirNode(input_size, 200)
    readout = Oger.nodes.RidgeRegressionNode(0)

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout])

    # Nested dictionary
    gridsearch_parameters = {reservoir:{'input_scaling': mdp.numx.arange(0.1, 1, .2), 'spectral_radius':mdp.numx.arange(0.1, 1.5, .3)}}

    # Instantiate an optimizer
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)
    
    # Do the grid search
    opt.grid_search(data, flow, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=5)

    # Get the minimal error
    min_error, parameters = opt.get_minimal_error()
    
    print 'The minimal error is ' + str(min_error)
    print 'The corresponding parameter values are: '
    output_str = '' 
    for node in parameters.keys():
        for node_param in parameters[node].keys():
            output_str += str(node) + '.' + node_param + ' : ' + str(parameters[node][node_param]) + '\n'
    print output_str
