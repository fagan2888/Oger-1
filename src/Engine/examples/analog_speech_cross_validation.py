'''
Created on Dec 4, 2009

@author: dvrstrae
'''

from Engine import datasets
from Engine import error_measures
from Engine import reservoir_nodes
from Engine import utility_nodes
from Engine import crossvalidation

import mdp

from Engine.linear_nodes import RidgeRegressionNode

if __name__ == "__main__":
    '''
        Simple example to show cross_validation on an isolated digit recognition task (subset of TI46 corpus).
    '''
    nx, ny = 4,1
    washout=0
    train_frac = .9;

    [x,y] = datasets.analog_speech(indir="/Users/dvrstrae/restored/Lyon128")
    
    n_samples = len(x);

    inputs = x[0].shape[1]
    # construct individual nodes
    reservoir = reservoir_nodes.ReservoirNode(inputs, 100, input_scaling = 1)
    readout = RidgeRegressionNode()
    mnnode = utility_nodes.MeanAcrossTimeNode()
    wtanode = utility_nodes.WTANode();

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout], verbose=1)
    flownode = mdp.hinet.FlowNode(flow)
    
    print "Cross_validation..."
    # train and test it
    errors = crossvalidation.cross_validate(x,y,flownode, error_measures.wer, 10)
    print "Mean word error rate across folds: " + str(mdp.numx.mean(errors))
