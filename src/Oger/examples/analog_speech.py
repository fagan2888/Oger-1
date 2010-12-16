import Oger
import mdp
import pylab
import scipy as sp

if __name__ == "__main__":

    n_subplots_x, n_subplots_y = 2, 1
    train_frac = .9

    [inputs, outputs] = Oger.datasets.analog_speech(indir="../datasets/Lyon_decimation_128")
    
    n_samples = len(inputs)
    n_train_samples = int(round(n_samples * train_frac))
    n_test_samples = int(round(n_samples * (1 - train_frac)))
    
    input_dim = inputs[0].shape[1]

    # construct individual nodes
    reservoir = Oger.nodes.LeakyReservoirNode(input_dim=input_dim, output_dim=100, input_scaling=1, leak_rate=0.1)
    readout = Oger.nodes.RidgeRegressionNode(0.001)
    
    gridsearch_params = {readout:{'ridge_param':mdp.numx.power(10., mdp.numx.arange(-15, 0, .5))}}
    cross_validate_function = Oger.evaluation.n_fold_random
    error_measure = Oger.utils.nrmse
    n_folds = 5
    Oger.utils.optimize_parameters(Oger.nodes.RidgeRegressionNode, gridsearch_parameters=gridsearch_params, cross_validate_function=cross_validate_function, error_measure=error_measure, n_folds=5)
#    
    mnnode = Oger.nodes.MeanAcrossTimeNode()

    # build network with MDP framework
    flow = Oger.nodes.InspectableFlow([reservoir, readout, mnnode])
        
    pylab.subplot(n_subplots_x, n_subplots_y, 1)
    pylab.imshow(inputs[0].T, aspect='auto', interpolation='nearest')
    pylab.title("Cochleogram (input to reservoir)")
    pylab.ylabel("Channel")
    
    
    print "Training..."
    # train and test it
    flow.train([[], \
                zip(inputs[0:n_train_samples - 1], \
                    outputs[0:n_train_samples - 1]), \
                []])

    ytrain, ytest = [], []
    print "Applying to trainingset..."
    for xtrain in inputs[0:n_train_samples - 1]:
        ytrain.append(flow(xtrain))
    print "Applying to testset..."
    for xtest in inputs[n_train_samples:]:
        ytest.append(flow(xtest))
        
    pylab.subplot(n_subplots_x, n_subplots_y, 2)
    pylab.plot(flow.inspect(reservoir))
    pylab.title("Sample reservoir states")
    pylab.xlabel("Timestep")
    pylab.ylabel("Activation")
    
    #pylab.show()
    ymean = sp.array([sp.argmax(sample) for sample in 
                      outputs[n_train_samples:]])
    ytestmean = sp.array([sp.argmax(sample) for sample in ytest])
    
    print "Error: " + str(mdp.numx.mean(Oger.utils.loss_01(ymean,
                                                               ytestmean)))
    pylab.show()
    
