import Oger
import mdp
import pylab
import scipy as sp
import brian




if __name__ == "__main__":

    n_subplots_x, n_subplots_y = 2, 1
    train_frac = .9
    
    [inputs, outputs] = Oger.datasets.analog_speech(indir="/Users/dvrstrae/Documents/workspace/organic-reservoir-computing-engine/src/Oger/datasets/Lyon_128")
    
    n_samples = len(inputs)
    n_train_samples = int(round(n_samples * train_frac))
    n_test_samples = int(round(n_samples * (1 - train_frac)))
    
    input_dim = inputs[0].shape[1]
    
    reservoir = Oger.nodes.BrianIFReservoirNode(input_dim, output_dim=400, dtype=float)
    readout = Oger.nodes.RidgeRegressionNode(0.001)
    mnnode = Oger.nodes.MeanAcrossTimeNode()
    
    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout, mnnode])
    
#    pylab.subplot(n_subplots_x, n_subplots_y, 1)
#    pylab.imshow(inputs[0].T, aspect='auto', interpolation='nearest')
#    pylab.title("Cochleogram (input to reservoir)")
#    pylab.ylabel("Channel")
#    
    
    print "Training..."
    # train and test it
    flow.train([[inputs[0:n_train_samples - 1]], \
                zip(inputs[0:n_train_samples - 1], \
                    outputs[0:n_train_samples - 1]), \
                [None]])
    
    ytrain, ytest = [], []
    print "Applying to trainingset..."
    for xtrain in inputs[0:n_train_samples - 1]:
        ytrain.append(flow(xtrain))
    print "Applying to testset..."
    for xtest in inputs[n_train_samples:]:
        ytest.append(flow(xtest))
    
    pylab.subplot(n_subplots_x, n_subplots_y, 2)
    pylab.plot(reservoir.states)
    pylab.title("Sample reservoir states")
    pylab.xlabel("Timestep")
    pylab.ylabel("Activation")
    
    ymean = sp.array([sp.argmax(sample) for sample in 
                      outputs[n_train_samples:]])
    ytestmean = sp.array([sp.argmax(sample) for sample in ytest])
    
    print "Error: " + str(mdp.numx.mean(Oger.utils.loss_01(ymean,
                                                               ytestmean)))
    pylab.show()