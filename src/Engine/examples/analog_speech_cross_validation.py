'''
Created on Dec 4, 2009

@author: dvrstrae
'''

'''
Created on Aug 24, 2009

@author: dvrstrae
'''
from Engine import datasets
from Engine import error_measures
from Engine import reservoir_nodes
from Engine import utility_nodes
from Engine import crossvalidation

import mdp
import pylab
#import scipy as sp

from Engine.linear_nodes import RidgeRegressionNode

if __name__ == "__main__":

    nx, ny = 4,1
    washout=0
    train_frac = .9;

    [x,y] = datasets.analog_speech(indir="/Users/dvrstrae/Lyon128")
    
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
    
    crossvalidation.cross_validate(x,y,flownode, error_measures.wer, 10)
    
    pylab.subplot(nx,ny,1)
    pylab.imshow(x[0].T,aspect='auto', interpolation='nearest')
    pylab.title("Cochleogram (input to reservoir)")
    pylab.ylabel("Channel")
    
    
    print "Cross_validation..."
    # train and test it
    
