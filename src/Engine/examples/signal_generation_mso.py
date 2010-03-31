import mdp
import pylab
import Engine

if __name__ == "__main__":

    freerun_steps = 1000
    
    x = Engine.datasets.mso(sample_len=10000)
    xtrain = x[0][0:-freerun_steps]
    ytrain = xtrain[1:]
    xtrain = xtrain[0:-1]

    # construct individual nodes
    reservoir = Engine.nodes.FeedbackReservoirNode(output_dim=100, input_scaling=0.1)    
    Engine.nodes.MixIn(Engine.nodes.FeedbackReservoirNode, Engine.nodes.LeakyReservoirNode)
    reservoir.leak_rate = 0.9

    readout = Engine.nodes.RidgeRegressionNode(0.)
    fb = Engine.nodes.FeedbackNode(n_timesteps=freerun_steps)

    # build network with MDP framework
    flow = Engine.nodes.InspectableFlow([reservoir, readout, fb], verbose=1)
        
    data = [ [xtrain, fb] , [[xtrain, ytrain], None], [xtrain, [None, x[0][-freerun_steps:]]]];
    
    # Nested dictionary
    gridsearch_parameters = {readout:{'ridge_param': mdp.numx.exp(mdp.numx.arange(-16, 2, 1))}}

    # Instantiate an optimizer
    opt = Engine.evaluation.Optimizer(gridsearch_parameters, Engine.utils.timeslice(range(50,500), Engine.utils.mse))
    
    # Do the grid search
    opt.grid_search(data, flow, cross_validate_function=Engine.evaluation.train_test_only, training_fraction=0.5, random=False)

    # Get the minimal error
    min_error, parameters = opt.get_minimal_error()
    print parameters
    
    # TODO: this should be a dict of dicts, same as gridsearch_parameters
    readout.ridge_param = parameters[readout]["ridge_param"]
    
    data = [ [xtrain, ] , [[xtrain, ytrain], ], [xtrain, ]];

    # Train the flow
    flow.train(data)
                
    # Execute the flow on the training data
    reservoir.reset_states = True
    flow.execute(xtrain)
    
    # Save the states of the reservoir during training for later plotting
    training_states = flow.inspect(reservoir)

    # Without this line, we start from the estimated last timestep, while
    # this line starts the feedback node from the actual last timestep.
    fb.last_value = mdp.numx.atleast_2d(ytrain[-1,:]) 
    
    # Turn of the initialization of the reservoir to zero, needed for freerun
    reservoir.reset_states = False
    
    # Execute the flow using the feedbacknode as input
    fb.reset()
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
    pylab.plot(x[0][-2 * freerun_steps:])
    pylab.show()