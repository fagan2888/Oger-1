'''
Created on Feb 24, 2010

@author: dvrstrae
'''
'''
Created on Aug 24, 2009

@author: dvrstrae
'''
import datasets
import reservoir_nodes
import linear_nodes
import utility_nodes
import mdp
import pylab


if __name__ == "__main__":

    freerun_steps = 500
    
    x = datasets.mackey_glass(sample_len=10000)
    xtrain = x[0:-freerun_steps]
    ytrain = xtrain[1:]
    xtrain = xtrain[0:-1]
        
    # construct individual nodes
    reservoir = reservoir_nodes.LeakyReservoirNode(output_dim=100, input_scaling=1, leak_rate=0.5)
    readout = linear_nodes.RidgeRegressionNode(0.01)
    fb = FeedbackNode(n_timesteps=freerun_steps)
    
     
    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout, fb], verbose=1)
        
    # Train the reservoir to do one-step ahead prediction using the teacher-forced signal
    flow.train([ [[xtrain]] , [[xtrain, ytrain]], [[xtrain]]])
                
    # Save the states of the reservoir during training for later plotting
    training_states = reservoir.states
    
    # Turn of the initialization of the reservoir to zero, needed for freerun
    reservoir.reset_states = False
    
    # Execute the flow using the feedbacknode as input
    output = flow.execute(fb)
        
    #collected_states=mdp.numx.array(reservoir.collected_states)
    pylab.subplot(311)
    #pylab.plot(mdp.numx.concatenate((mdp.numx.array(training_states[-freerun_steps:,:]), collected_states[0:freerun_steps,:])))
    pylab.plot(training_states[-freerun_steps:, :])
    
    pylab.subplot(312)
    #pylab.plot(reservoir.states[-2*freerun_steps:])
    #pylab.plot(collected_states[0:freerun_steps,:])
    
    pylab.subplot(313)
    pylab.plot(mdp.numx.concatenate((ytrain[-freerun_steps:], output[0:freerun_steps])))
    pylab.plot(x[-2 * freerun_steps:])
