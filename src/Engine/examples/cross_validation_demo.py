from Engine import datasets 
from Engine import error_measures
from Engine import reservoir_nodes
from Engine import crossvalidation

import mdp

from Engine.linear_nodes import RidgeRegressionNode

if __name__ == "__main__":
    n_inputs = 1
    system_order = 30;
    washout=30;

    [inputs,outputs] = datasets.narma30()

    # construct individual nodes
    reservoir = reservoir_nodes.ReservoirNode(output_dim = 100)
    readout = RidgeRegressionNode(0)

    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout], verbose=1)
    RC = mdp.hinet.FlowNode(flow)
   
   
    errors = crossvalidation.cross_validate(inputs, outputs, RC, error_measures.nrmse, 10)
    
    print "Mean error: " + str(mdp.numx.mean(errors))
    print errors
