import Oger
import pylab
import mdp


if __name__ == "__main__":

    freerun_steps = 500
    
    x = Oger.datasets.mackey_glass(sample_len=10000)
    xtrain = x[0][0:-freerun_steps]
    ytrain = xtrain[1:]
    xtrain = xtrain[0:-1]
        
    # construct individual nodes
    reservoir = Oger.nodes.FeedbackReservoirNode(output_dim=100, input_scaling=.8)    
    Oger.utils.mix_in(Oger.nodes.FeedbackReservoirNode, Oger.nodes.LeakyReservoirNode)
    reservoir.leak_rate = 0.5
    
    readout = Oger.nodes.RidgeRegressionNode(0.0001)
    fb = Oger.nodes.FeedbackNode(n_timesteps=freerun_steps)
         
    # build network with MDP framework
    flow = Oger.nodes.InspectableFlow([reservoir, readout, fb], verbose=1)
        
    # Train the reservoir to do one-step ahead prediction using the teacher-forced signal
    flow.train([ [[]] , [[xtrain, ytrain]], [[]]])
                    
    # Save the states of the reservoir during training for later plotting
    flow.execute(xtrain)
    training_states = flow.inspect(reservoir)
    training_output = flow.inspect(readout)
     
    reservoir.reset_states = False
    
    # Without this line, we start from the estimated last timestep, while
    # this line starts the feedback node from the actual last timestep.
    fb.last_value = mdp.numx.atleast_2d(ytrain[-1, :]) 

    # Execute the flow using the feedbacknode as input
    output = flow.execute(fb)
        
    pylab.subplot(211)
    pylab.plot(mdp.numx.concatenate((training_states[-freerun_steps:], flow.inspect(reservoir)[0:freerun_steps])))
    
    pylab.subplot(212)
    pylab.plot(mdp.numx.concatenate((ytrain[-freerun_steps:], output[0:freerun_steps])))
    pylab.plot(x[0][-2 * freerun_steps:])
    
    pylab.show()
