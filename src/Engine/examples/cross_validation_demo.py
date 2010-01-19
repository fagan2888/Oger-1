from Engine import datasets 
from Engine import error_measures
from Engine import reservoir_nodes

import mdp

from Engine.linear_nodes import RidgeRegressionNode

if __name__ == "__main__":
    n_inputs = 1
    system_order = 30;
    washout=30;

    [inputs,outputs] = datasets.narma30(sample_len=9000)

    # construct individual nodes
    reservoir = reservoir_nodes.ReservoirNode(n_inputs,100)
    readout = RidgeRegressionNode(0)

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout], verbose=1)
   
    # train and test it
    for x,y in zip(inputs[0:8], outputs[0:8]): 
        flow.train([x,[(x,y)]])
        
    testout = flow(inputs[9])
    
    print error_measures.nrmse(outputs[9],testout, discard=washout)
    print "finished"