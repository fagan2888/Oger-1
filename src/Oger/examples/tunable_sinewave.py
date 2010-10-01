import Oger
import pylab
import mdp

def sinewaves(freqs, dt):
    t = mdp.numx.atleast_2d(mdp.numx.arange(freqs.shape[0])).T
    return(mdp.numx.atleast_2d(mdp.numx.sin(2 * mdp.numx.pi * freqs * t * dt)))
    
if __name__ == "__main__":
    N = 200
    
    # seconds per simulation timestep
    dt = .01

    freqtrain = mdp.numx.vstack([mdp.numx.hstack((mdp.numx.atleast_2d(mdp.numx.arange(1, 0, -.002)), mdp.numx.atleast_2d(mdp.numx.arange(0, 1, .002)))).T] * 4)
    freqtest = mdp.numx.atleast_2d(mdp.numx.sin(mdp.numx.arange(mdp.numx.pi / 2, 6, .002))).T / 2 + .5
       
    x = sinewaves(freqtrain, dt)
    xtest = sinewaves(freqtest, dt)
    xtrain = x[0:-1, :] 
    ytrain = x[1:, :]

    freqtrain = freqtrain[0:-1, :]
    freerun_steps = xtest.shape[0]

    # construct individual nodes
    reservoir = Oger.nodes.FeedbackReservoirNode(output_dim=N, input_dim=2, input_scaling=1)
    Oger.utils.mix_in(Oger.nodes.FeedbackReservoirNode, Oger.nodes.LeakyReservoirNode)
    reservoir.leak_rate = 1    
    
    readout = Oger.nodes.RidgeRegressionNode(0.1)
    
    fb = Oger.nodes.FeedbackNode(n_timesteps=freerun_steps)
    
    # build network with MDP framework
    flow = Oger.nodes.InspectableFlow([reservoir, readout, fb], verbose=1)
    
    fb.last_value = mdp.numx.atleast_2d(freqtrain[0, :])
    
    # Train the reservoir to do one-step ahead prediction using the teacher-forced signal
    # An additional input giving the frequency of the desired sine wave is also given   
    flow.train([ [] , Oger.utils.ConcatenatingIterator([xtrain, freqtrain], ytrain) , []])
    #flow.train([ [] , [[mdp.numx.hstack((xtrain, freqtrain)), ytrain]], []])
                    
    # Save the states of the reservoir during training for later plotting
    flow.execute(Oger.utils.ConcatenatingIterator([xtrain, freqtrain]))
    training_states = flow.inspect(reservoir)
    training_output = flow.inspect(readout)
     
    reservoir.reset_states = False
    
    # Without this line, we start from the estimated last timestep, while
    # this line starts the feedback node from the actual last timestep.
    fb.last_value = mdp.numx.atleast_2d(ytrain[-1, :]) 

    fb.reset()
    # Execute the flow using the feedbacknode as input
    ci = Oger.utils.ConcatenatingIterator([fb, freqtest])
    test_output = flow.execute(ci)
        
    #collected_states=mdp.numx.array(reservoir.collected_states)
    pylab.subplot(311)
    pylab.plot(mdp.numx.concatenate((training_states, flow.inspect(reservoir))))
    
    pylab.subplot(312)
    pylab.plot(mdp.numx.concatenate((freqtrain, freqtest)))
    
    pylab.subplot(313)
    pylab.plot(mdp.numx.concatenate((xtrain, xtest)), 'b')
    pylab.plot(mdp.numx.concatenate((training_output, test_output)), 'r')

    pylab.show()
