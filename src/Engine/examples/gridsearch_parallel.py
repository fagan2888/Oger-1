import mdp
import Engine
import mdp.parallel.pp_support
import pp

if __name__ == '__main__':
    ''' Example of doing a grid-search
        Runs the NARMA 30 task for bias values = 0 to 2 with 0.5 stepsize and spectral radius = 0.1 to 1 with stepsize 0.5
    '''
    input_size = 1
    inputs, outputs = Engine.datasets.narma30()

    data = [inputs, zip(inputs, outputs)]

    # construct individual nodes
    reservoir = Engine.nodes.ReservoirNode(input_size, 200)
    readout = Engine.nodes.RidgeRegressionNode(0)

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout])

    # Nested dictionary
    gridsearch_parameters = {reservoir:{'input_scaling': mdp.numx.arange(0.1, 1, 0.2), 'spectral_radius':mdp.numx.arange(0.1, 1.5, 0.3)}}

    # Instantiate an optimizer
    opt = Engine.evaluation.Optimizer(gridsearch_parameters, Engine.utils.nrmse)
    
    mdp.activate_extension("parallel")
#    opt.scheduler = mdp.parallel.ProcessScheduler(n_processes=4, verbose=True)
#    opt.scheduler = mdp.parallel.ThreadScheduler(n_threads=4, verbose=True)
#    opt.scheduler = mdp.parallel.pp_support.LocalPPScheduler(ncpus=2, max_queue_length=0, verbose=True)

    job_server = pp.Server(0, ppservers=("clsnn001:60000",))
    opt.scheduler = mdp.parallel.pp_support.PPScheduler(ppserver=job_server, verbose=True)
    
    # Do the grid search
    opt.grid_search(data, flow, cross_validate_function=Engine.evaluation.n_fold_random, n_folds=5)

    # Get the minimal error
    min_error, parameters = opt.get_minimal_error()
    
    print 'The minimal error is ' + str(min_error)
    print 'The corresponding parameter values are: '
    output_str = '' 
    for node in parameters.keys():
        for node_param in parameters[node].keys():
            output_str += str(node) + '.' + node_param + ' : ' + str(parameters[node][node_param]) + '\n'
    print output_str
