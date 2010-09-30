import Engine
import pylab
import mdp

if __name__ == "__main__":
    """ This example shows how to perform internal optimization of one or multiple parameters of a node during training. 
        This can be used to e.g. optimize the ridge parameter of the ridge regression node as is done here.
    """
    inputs = 1
    timesteps = 10000
    washout = 30

    nx = 4
    ny = 1

    [x, y] = Engine.datasets.narma30(sample_len=1000)

    # construct individual nodes
    reservoir = Engine.nodes.ReservoirNode(inputs, 100, input_scaling=0.05)
    readout = Engine.nodes.RidgeRegressionNode()
    
    gridsearch_params = {readout:{'ridge_param':mdp.numx.power(10., mdp.numx.arange(-15, 0, .5))}}
    cross_validate_function = Engine.evaluation.n_fold_random
    error_measure = Engine.utils.nrmse
    n_folds = 5
    Engine.utils.optimize_parameters(Engine.nodes.RidgeRegressionNode, gridsearch_parameters=gridsearch_params, cross_validate_function=cross_validate_function, error_measure=error_measure, n_folds=5)
    
    
    # build network with MDP framework
    flow = Engine.nodes.InspectableFlow([reservoir, readout], verbose=1)
    
    data = [[], zip(x[0:-1], y[0:-1])]
    
    # train the flow 
    flow.train(data)
    
    #apply the trained flow to the training data and test data
    trainout = flow(x[0])
    testout = flow(x[9])

    print "NRMSE: ", Engine.utils.nrmse(y[9], testout)
    print "Optimal ridge parameter : ", readout.ridge_param
    #plot the input
    pylab.subplot(nx, ny, 1)
    pylab.plot(x[0])
    
    #plot everything
    pylab.subplot(nx, ny, 2)
    pylab.plot(trainout, 'r')
    pylab.plot(y[0], 'b')

    pylab.subplot(nx, ny, 3)
    pylab.plot(testout, 'r')
    pylab.plot(y[9], 'b')
    
    pylab.subplot(nx, ny, 4)
    pylab.plot(flow.inspect(reservoir))
    pylab.show()
    
