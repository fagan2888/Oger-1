'''
Created on Aug 24, 2009

@author: dvrstrae
'''
import datasets
import error_measures
import reservoir_nodes
import utility_nodes

import mdp
from scipy import io
import pylab
import scipy as sp

from Engine.linear_nodes import RidgeRegressionNode

if __name__ == "__main__":

    nx, ny = 2,1
    train_frac = .9;

    [x,y] = datasets.analog_speech(indir="/Users/dvrstrae/Lyon128")
    
    n_samples = len(x);
    n_train_samples = int(round(n_samples*train_frac));
    n_test_samples = int(round(n_samples*(1-train_frac)));
    
    input_dim = x[0].shape[1]

    # construct individual nodes
    reservoir = reservoir_nodes.ReservoirNode(input_dim, 100, input_scaling = 1)
    readout = RidgeRegressionNode()
    mnnode = utility_nodes.MeanAcrossTimeNode()

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout, mnnode])
    
    pylab.subplot(nx,ny,1)
    pylab.imshow(x[0].T,aspect='auto', interpolation='nearest')
    pylab.title("Cochleogram (input to reservoir)")
    pylab.ylabel("Channel")
    
    
    print "Training..."
    # train and test it
    flow.train([ [x[0:n_train_samples-1]],zip(x[0:n_train_samples-1], y[0:n_train_samples-1]), [None] ])

    ytrain,ytest=[],[]
    print "Applying to trainingset..."
    for xtrain in mdp.utils.progressinfo(x[0:n_train_samples-1]):
        ytrain.append(flow(xtrain))
    print "Applying to testset..."
    for xtest in mdp.utils.progressinfo(x[n_train_samples:]):
        ytest.append(flow(xtest))
        
    pylab.subplot(nx,ny,2)
    pylab.plot(reservoir.states)
    pylab.title("Sample reservoir states")
    pylab.xlabel("Timestep")
    pylab.ylabel("Activation")
    pylab.show()
    
    #pylab.show()
    ymean=sp.array([sp.argmax(sp.mean(sample, axis=0)) for sample in y[n_train_samples:]])
    ytestmean=sp.array([sp.argmax(sp.mean(sample, axis=0)) for sample in ytest])
    
    print mdp.numx.mean(error_measures.loss_01(ymean, ytestmean))
    print "finished"