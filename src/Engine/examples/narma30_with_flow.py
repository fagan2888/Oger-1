'''
Created on Mar 8, 2010

@author: dvrstrae
'''
from mdp.hinet.flownode import FlowNode
import datasets 
import error_measures
import reservoir_nodes
import linear_nodes
import pylab
import mdp


if __name__ == "__main__":

    inputs = 1
    timesteps = 10000
    washout=30

    nx=4
    ny=1

    [x,y] = datasets.narma30(sample_len=1000)

    # construct individual nodes
    reservoir = reservoir_nodes.ReservoirNode(inputs,100)
    readout = linear_nodes.RidgeRegressionNode(0)

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout], verbose=1)
    RC = mdp.hinet.FlowNode(flow)
    
    pylab.subplot(nx,ny,1)
    pylab.plot(x[0])
    
    # train the flow 
    flow.train([ [[x[0]]], [[x[0],y[0]]] ])
    #RC.train(x[0], y[0])
    
    #flow.stop_training()
    
    #apply the trained flow to the training data and test data
    trainout = flow(x[0])
    testout = flow(x[1])
    
    #plot everything
    pylab.subplot(nx,ny,2)
    pylab.plot(trainout, 'r')
    pylab.plot(y[0], 'b')

    pylab.subplot(nx,ny,3)
    pylab.plot(testout[washout:], 'r')
    pylab.plot(y[1][washout:], 'b')
    
    pylab.subplot(nx,ny,4)
    pylab.plot(reservoir.states)
    pylab.show()
    
    print error_measures.nrmse(y[1],testout, discard=washout)
    print "finished"

