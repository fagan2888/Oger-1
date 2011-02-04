import mdp
import pylab
import Oger

if __name__ == "__main__":

    mdp.numx.seterr(all='raise')
    mdp.numx.random.seed(1234)
    freerun_steps = 1000
    
    x = Oger.datasets.mso(sample_len=50000)
    xtrain = x[0][0:-freerun_steps]
    ytrain = xtrain[1:]
    xtrain = xtrain[0:-1]
    ytest = x[0][-freerun_steps:]

    # construct individual nodes
    reservoir = Oger.nodes.FeedbackReservoirNode(output_dim=100, input_scaling=1)    
    Oger.utils.mix_in(Oger.nodes.FeedbackReservoirNode, Oger.nodes.LeakyReservoirNode)
    reservoir.leak_rate = 0.1

    readout = Oger.nodes.RidgeRegressionNode()
    Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode, 100)
    
    fb = Oger.nodes.FeedbackNode(n_timesteps=freerun_steps)

    # build network with MDP framework
    flow = Oger.nodes.InspectableFlow([reservoir, readout, fb], verbose=1)
 
  
#    gridsearch_params = {readout:{'ridge_param':mdp.numx.power(10., mdp.numx.arange(-16, 2, 1))}}
#    cross_validate_function = Oger.evaluation.train_test_only
#    training_fraction = 0.5
#    error_measure = Oger.utils.timeslice(range(50, 500), Oger.utils.mse)
#    Oger.utils.optimize_parameters(Oger.nodes.RidgeRegressionNode, gridsearch_parameters=gridsearch_params, cross_validate_function=cross_validate_function, error_measure=error_measure, training_fraction=training_fraction)
#    
    data = [ [] , [[xtrain, ytrain]], [[xtrain, ytest]]]

    Oger.evaluation.validate_freerun(data, flow, Oger.utils.nrmse)
    
    # Turn of the initialization of the reservoir to zero, needed for freerun
    reservoir.reset_states = False
    
    # Execute the flow using the feedbacknode as input
    freerun_states = flow.inspect(reservoir)
    
    #collected_states=mdp.numx.array(reservoir.collected_states)
#    pylab.subplot(311)
#    #pylab.plot(mdp.numx.concatenate((mdp.numx.array(training_states[-freerun_steps:,:]), collected_states[0:freerun_steps,:])))
#    pylab.plot(training_states[-freerun_steps:, :])
#    
#    pylab.subplot(312)
#    pylab.plot(freerun_states)
#    #pylab.plot(reservoir.states[-2*freerun_steps:])
#    #pylab.plot(collected_states[0:freerun_steps,:])
#    
#    pylab.subplot(313)
#    pylab.plot(mdp.numx.concatenate((ytrain[-freerun_steps:], output[0:freerun_steps])))
#    pylab.plot(x[0][-2 * freerun_steps:])
#    pylab.show()
