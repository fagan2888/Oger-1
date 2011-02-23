import Oger
import scipy as sp
import time
import mdp.parallel

if __name__ == '__main__':
    ''' Example of doing a grid-search
        Runs the NARMA 30 task for input scaling values = 0.1 to 1 with 0.2 stepsize and spectral radius = 0.1 to 1.5 with stepsize 0.3
    '''
    input_size = 1
    inputs, outputs = Oger.datasets.narma30()

    data = [[], zip(inputs, outputs)]

    # construct individual nodes
    reservoir = Oger.nodes.ReservoirNode(input_size, 100)
    readout = Oger.nodes.RidgeRegressionNode()

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout])

    # Nested dictionary
    # For cma_es, each parameter 'range' consists of an initial value and a standard deviation
    gridsearch_parameters = {reservoir:{'input_scaling': mdp.numx.array([0.3, .5]), 'spectral_radius':mdp.numx.array([.9, .5])}}

    # Instantiate an optimizer
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)

    print 'Sequential execution...'
    start_time = time.time()
    opt.cma_es(data, flow, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=5, options={'maxiter':5, 'bounds':[0.01, None], 'seed':1234})
    seq_duration = int(time.time() - start_time)
    print 'Duration: ' + str(seq_duration) + 's'

    # Get the optimal flow and run cross-validation with it 
    opt_flow = opt.get_optimal_flow()

    print 'Performing cross-validation with the optimal flow.'
    errors = Oger.evaluation.validate(data, opt_flow, Oger.utils.nrmse, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=5, progress=False)
    print 'Mean error over folds: ' + str(sp.mean(errors))

    # Do the grid search
    print 'Parallel execution...'
    opt.scheduler = mdp.parallel.ProcessScheduler(n_processes=2)
    mdp.activate_extension("parallel")
    start_time = time.time()
    opt.cma_es(data, flow, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=5, options={'maxiter':5, 'bounds':[0.01, None], 'seed':1234})
    par_duration = int(time.time() - start_time)
    print 'Duration: ' + str(par_duration) + 's'

    print 'Speed up factor: ' + str(float(seq_duration) / par_duration)

    # Get the optimal flow and run cross-validation with it 
    opt_flow = opt.get_optimal_flow()

    print 'Performing cross-validation with the optimal flow. Note that this result can differ slightly from the one above because of different choices of randomization of the folds.'

    errors = Oger.evaluation.validate(data, opt_flow, Oger.utils.nrmse, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=5, progress=False)
    print 'Mean error over folds: ' + str(sp.mean(errors))
