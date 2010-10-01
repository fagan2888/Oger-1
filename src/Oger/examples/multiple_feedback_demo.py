import Oger
import pylab
import mdp


if __name__ == "__main__":
    '''This demonstrates the use of feeding back two different values using the same feedback node:
    
            |--------------------------------------|
            ->   _____________    ->  Readout 1 -->|------|
                | Reservoir  | --|                        |--> Readout 3
            ->  -------------     ->  Readout 2 -->|------|
            |--------------------------------------|
            
        The readouts are trained using different values of the regularization constant, so they yield 
        different outputs, but are still fed back into the same reservoir. Their outputs are then fed to a 
        third ridge regression node. 
        This example is perhaps not very practical, but it shows how more complex feedback schemes can be built.
    '''
    freerun_steps = 500
    
    N = 100;
    x = Oger.datasets.mackey_glass(sample_len=10000)
    xtrain = x[0][0:-freerun_steps]
    ytrain = xtrain[1:]
    
    #The reservoir expects two-dimensional input, so we just duplicate the teacher signal
    xtrain = mdp.numx.hstack((xtrain[0:-1], xtrain[0:-1]))
        
    # construct individual nodes
    reservoir = Oger.nodes.FeedbackReservoirNode(output_dim=N, input_dim=2, input_scaling=1)    
    Oger.utils.mix_in(Oger.nodes.FeedbackReservoirNode, Oger.nodes.LeakyReservoirNode)
    reservoir.leak_rate = 0.5    
    
    readout = Oger.nodes.RidgeRegressionNode(0.0001, input_dim=N)
    readout2 = Oger.nodes.RidgeRegressionNode(0.0002, input_dim=N) 
    
    output_layers = mdp.hinet.SameInputLayer([readout, readout2])
    fb = Oger.nodes.FeedbackNode(n_timesteps=freerun_steps, input_dim=2)


    readout3 = Oger.nodes.RidgeRegressionNode(0.000001, input_dim=2)
    # build network with MDP framework
    flow = Oger.nodes.InspectableFlow([reservoir, output_layers, fb, readout3], verbose=1)
        
    # Train the reservoir to do one-step ahead prediction using the teacher-forced signal
    flow.train([ [] , [[xtrain, ytrain]], [[xtrain]], [[xtrain, ytrain]]])
                    
    # Save the states of the reservoir during training for later plotting
    y = flow.execute(xtrain)
    training_states = flow.inspect(reservoir)
    training_output = flow.inspect(output_layers)
     
    pylab.figure()
    pylab.plot(y)
    reservoir.reset_states = False
    
    # Without this line, we start from the estimated last timestep, while
    # this line starts the feedback node from the actual last timestep.
    fb.last_value = mdp.numx.atleast_2d(xtrain[-1, :]) 

    # Execute the flow using the feedbacknode as input
    output = flow.execute(fb)
        
    #collected_states=mdp.numx.array(reservoir.collected_states)
    pylab.subplot(311)
    #pylab.plot(mdp.numx.concatenate((mdp.numx.array(training_states[-freerun_steps:,:]), collected_states[0:freerun_steps,:])))
    pylab.plot(training_states[-freerun_steps:, :])
    
    pylab.subplot(312)
    pylab.plot(mdp.numx.concatenate((training_states[-freerun_steps:], flow.inspect(reservoir)[0:freerun_steps])))
    
    pylab.subplot(313)
    pylab.plot(mdp.numx.concatenate((xtrain[-freerun_steps:, 1], output[0:freerun_steps, 0])))
    pylab.plot(x[0][-2 * freerun_steps:])
    pylab.plot(training_output[-freerun_steps:])

    pylab.show()
