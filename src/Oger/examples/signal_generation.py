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
    #
    # The second item in the list below is used to train the readout to do one step ahead prediction,
    # and the third item in the list is used to warmup the reservoir and initialize the FeedbackNode to the correct
    # (teacher forced) initial value
    
    flow.train([ [] , [[xtrain, ytrain]], [[xtrain, ytrain]]])
                    
    # The inspection field of the reservoir contains a list of 2 arrays, since the reservoir was already run twice
    training_states = flow.inspect(reservoir)[1]
    
    # The inspection field of the readout only contains one array, since the readout was only executed once yet, during training of the feedbacknode
    training_output = flow.inspect(readout)
     
    reservoir.reset_states = False
    
    # Execute the flow using the feedbacknode as input
    output = flow.execute(fb)
        
    pylab.subplot(211)
    pylab.plot(mdp.numx.concatenate((training_states[-freerun_steps:], flow.inspect(reservoir)[0:freerun_steps])))
    
    pylab.subplot(212)
    pylab.plot(mdp.numx.concatenate((ytrain[-freerun_steps:], output[0:freerun_steps])))
    pylab.plot(x[0][-2 * freerun_steps:])
    
    pylab.show()
