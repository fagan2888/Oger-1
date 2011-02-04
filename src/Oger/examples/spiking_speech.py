import Oger
import mdp
import pylab
import scipy as sp
    
if __name__ == "__main__":

    n_subplots_x, n_subplots_y = 2, 1
    train_frac = .9
    
    [inputs, outputs] = Oger.datasets.analog_speech(indir="Lyon128")
    
    n_samples = len(inputs)
    n_train_samples = int(round(n_samples * train_frac))
    n_test_samples = int(round(n_samples * (1 - train_frac)))
    
    input_dim = inputs[0].shape[1]
    
    cell_params = {
            'tau_m'      : 20.0, # (ms)
            'tau_syn_E'  : 2.0, # (ms)
            'tau_syn_I'  : 4.0, # (ms)
            'tau_refrac' : 2.0, # (ms)
            'v_rest'     : 0.0, # (mV)
            'v_init'     : 0.0, # (mV)
            'v_reset'    : 0.0, # (mV)
            'v_thresh'   : 20.0, # (mV)
            'cm'         : 0.5}  # (nF)
    
    RateScale = 1e6
    Tstep = 10.0
    
    rngseed = 1240498
    
        
    
    # construct individual nodes
    reservoir = Oger.nodes.SpikingIFReservoirNode(input_dim, output_dim=400, dtype=float, exc_frac=0.8, exc_w=0.1, inh_w= -0.5, input_w=0.2, cell_params=cell_params,
                                       syn_delay=1.0, Cprob_exc=0.2, Cprob_inh=0.2, Cprob_inp=0.2, kernel=Oger.utils.exp_kernel(tau=30, dt=1), inp2spikes_conversion=Oger.utils.poisson_gen(rngseed))
    readout = Oger.nodes.RidgeRegressionNode(0.001)
    mnnode = Oger.nodes.MeanAcrossTimeNode()
    
    # build network with MDP framework
    flow = mdp.Flow([reservoir, readout, mnnode])
    
    pylab.subplot(n_subplots_x, n_subplots_y, 1)
    pylab.imshow(inputs[0].T, aspect='auto', interpolation='nearest')
    pylab.title("Cochleogram (input to reservoir)")
    pylab.ylabel("Channel")
    
    
    print "Training..."
    # train and test it
    flow.train([[], \
                zip(inputs[0:n_train_samples - 1], \
                    outputs[0:n_train_samples - 1]), \
                []])
    
    ytrain, ytest = [], []
    print "Applying to trainingset..."
    for xtrain in inputs[0:n_train_samples - 1]:
        ytrain.append(flow(xtrain))
    print "Applying to testset..."
    for xtest in inputs[n_train_samples:]:
        ytest.append(flow(xtest))
    
    pylab.subplot(n_subplots_x, n_subplots_y, 2)
    pylab.plot(reservoir.states)
    pylab.title("Sample reservoir states")
    pylab.xlabel("Timestep")
    pylab.ylabel("Activation")
    
    ymean = sp.array([sp.argmax(sample) for sample in 
                      outputs[n_train_samples:]])
    ytestmean = sp.array([sp.argmax(sample) for sample in ytest])
    
    print "Error: " + str(mdp.numx.mean(Oger.utils.loss_01(ymean,
                                                               ytestmean)))
    pylab.show()
