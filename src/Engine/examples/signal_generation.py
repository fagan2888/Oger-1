'''
Created on Feb 24, 2010

@author: dvrstrae
'''

import Engine
import pylab
import mdp


if __name__ == "__main__":

    freerun_steps = 500
    
    x = Engine.datasets.mackey_glass(sample_len=10000)
    xtrain = x[0:-freerun_steps - 1]
    ytrain = x[1:-freerun_steps]
        
    # construct individual nodes
    reservoir = Engine.nodes.LeakyReservoirNode(output_dim=100, input_scaling=1, leak_rate=0.5)
    readout = Engine.nodes.RidgeRegressionNode(0.01)
    fb = Engine.nodes.FeedbackNode(n_timesteps=freerun_steps)
    
     
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
    pylab.subplot(211)
    #pylab.plot(mdp.numx.concatenate((mdp.numx.array(training_states[-freerun_steps:,:]), collected_states[0:freerun_steps,:])))
    pylab.plot(training_states[-freerun_steps:, :])
        
    pylab.subplot(212)
    pylab.plot(mdp.numx.concatenate((ytrain[-freerun_steps:], output[0:freerun_steps])))
    pylab.plot(x[-2 * freerun_steps:])
    
    pylab.show()
