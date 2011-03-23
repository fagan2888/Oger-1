import Oger
import pylab
import mdp

if __name__ == "__main__":
    """ This example shows how to use several reservoirs in a layer and to perform forward input/reservoir selection 
    using a RidgeRegressionNode and the built-in select_features mix-in.
    """
    inputs = 1
    timesteps = 10000
    washout = 30

    nx = 3
    ny = 1

    [x, y] = Oger.datasets.narma30(sample_len=1000)

    # make a set of reservoirs
    reservoirs = []
    for i in range(25):
        reservoirs.append(Oger.nodes.ReservoirNode(inputs, 10, input_scaling=0.05))
    Res = mdp.hinet.SameInputLayer(reservoirs)

    readout = Oger.nodes.RidgeRegressionNode()

    # Find optimal ridge parameter
    gridsearch_params = {readout:{'ridge_param':mdp.numx.power(10., mdp.numx.arange(-15, 0, .5))}}
    cross_validate_function = Oger.evaluation.n_fold_random
    error_measure = Oger.utils.nrmse
    n_folds = 5
    #Oger.utils.optimize_parameters(Oger.nodes.RidgeRegressionNode, gridsearch_parameters=gridsearch_params, cross_validate_function=cross_validate_function, error_measure=error_measure, n_folds=n_folds)

    # Select a set of reservoirs as input for the RidgeRegressionNode
    Oger.utils.select_inputs(Oger.nodes.RidgeRegressionNode, n_inputs=len(reservoirs))

    # build network with MDP framework
    flow = Oger.nodes.InspectableFlow([Res, readout], verbose=1)

    data = [[], zip(x[0:-1], y[0:-1])]

    # train the flow 
    flow.train(data)

    #apply the trained flow to the training data and test data
    trainout = flow(x[0])
    testout = flow(x[-1])

    print "NRMSE: ", Oger.utils.nrmse(y[-1], testout)
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
    pylab.plot(y[-1], 'b')

    pylab.show()

