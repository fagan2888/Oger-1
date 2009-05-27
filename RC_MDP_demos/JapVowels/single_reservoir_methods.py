#===============================================================================
# Single Reservoir Methods for the Japanese Vowels Task
#===============================================================================

import shelve
import numpy as np
import aureservoir as au
import pylab
import random
import sys
sys.path.append("../")
from grhlib import errormeasures, filteresn


# VERSUCHE:
# - mit Filter: funktioniert nicht wirklich !
# - mit DS: auch net wirklich besser


def load_data():
    """ Loads the preprocessed training and test data. 
    """
    data = shelve.open("voweldata.dat")
    trainInputs = data["trainInputs"]
    trainOutputs = data["trainOutputs"]
    testInputs = data["testInputs"]
    testOutputs = data["testOutputs"]
    shifts = data["shifts"]
    data.close()
    return trainInputs, trainOutputs, testInputs, testOutputs, shifts

def analyze_data(data):
    """ analyzes characteristics of the (input) data
    """
    example = 4
    pylab.figure()
    pylab.subplot(2,1,1)
    pylab.plot(data[:,example])
    pylab.subplot(2,1,2)
    pylab.psd(data[:,example])

    example = 3
    pylab.figure()
    pylab.subplot(2,1,1)
    pylab.plot(data[:,example])
    pylab.subplot(2,1,2)
    pylab.psd(data[:,example])

    # nicht wirklich aussagekraeftig:
#    pylab.figure()
#    pylab.subplot(2,1,1)
#    pylab.plot(data.flatten())
#    pylab.subplot(2,1,2)
#    pylab.psd(data.flatten())
    
    pylab.show()
    exit(0)
    

def train_esn_nr1(train_in, train_out, model):
    # accumulate training vectors
    model.train_in = np.zeros((0,14))
    model.train_out = np.zeros((0,9))
    # make a random or a linear range for the loop
    model.train_looprange = range(len(train_in))
    if model.randrange:
        random.shuffle(model.train_looprange)

    for n in model.train_looprange:
        model.train_in = np.r_[model.train_in, train_in[n]]
        model.train_out = np.r_[model.train_out, train_out[n]]
    
    # analyze input data
#    analyze_data(model.train_in)
 
    # train the network
    model.setNoise( model.trainnoise )
    if model.ds==1:
        model.train(model.train_in.T.copy(), model.train_out.T.copy(), model.maxdelay+1)
    else:
        model.train(model.train_in.T.copy(), model.train_out.T.copy(), 0)
    model.setNoise( 0. )
    
    # print delay
    if model.ds == 1:
        model.delays = np.zeros((model.getOutputs(),model.getSize()+model.getInputs()))
        model.getDelays(model.delays)
        print "trained delays:"
        print model.delays[0]
        print model.delays[1]
        print model.delays[2]
        print "\tmean: ", model.delays.mean(), "\tmax: ", model.delays.max()
        print "output weights:"
        print "\tmean: ", model.getWout().mean(), "\tmax: ", \
              abs(model.getWout()).max()

def test_esn_nr1(test_in, model):
    # list for outputs
    test_out_anytime = []
    test_out_average = np.zeros((len(test_in),9))
    
    # make a random or a linear range for the loop
    model.test_looprange = range(len(test_in))
    if model.randrange:
        random.shuffle(model.test_looprange)

    # go through all test inputs
    for n in model.test_looprange:
        inputs = test_in[n].T.copy()
        outputs = np.zeros((9,inputs.shape[1]))
        #model.resetState()
        model.simulate(inputs, outputs)
        outputs = (outputs+0.9) / 1.8 # rescale signal into original scale 
        test_out_anytime.append( outputs.T )
    
    # average over the whole example
    for n in range( len(test_out_anytime) ):
        test_out_average[n] = test_out_anytime[n].mean(0)
    
    return test_out_anytime, test_out_average

def plot_results(show=0):
    """ Calculates NRMSEs and plots results
    """
    # first load the data
    data = shelve.open("vowelresults-nr1.dat")
    testout_anytime = data["testout_anytime"]
    testout_average = data["testout_average"]
    target_anytime = data["target_anytime"]
    target_average = data["target_average"]
    test_looprange = data["test_looprange"]
    data.close()

    # calc NRMSE for anytime results (accumulate vectors)
    testout_all = np.zeros((0,9))
    testout_last = np.zeros((370,9))
    target_all = np.zeros((0,9))
    targ_average = np.zeros((0,9))
    for n in range(len(testout_anytime)):
        testout_all = np.r_[testout_all, testout_anytime[n]]
        # calc also performance of the last output of an example
        testout_last[n] = testout_anytime[n][-1,:]
    # undo the possible random order of the examples
    for n in test_looprange:
        target_all = np.r_[target_all, target_anytime[n]]
        targ_average = np.r_[targ_average, target_average[n].reshape(1,-1)]
    
    nrmse_anytime = errormeasures.nrmse( testout_all.flatten(),
                                         target_all.flatten() )
    nrmse_average = errormeasures.nrmse( testout_average.flatten(),
                                         targ_average.flatten() )
    nrmse_last = errormeasures.nrmse( testout_last.flatten(),
                                      targ_average.flatten() )
    
    print "NRMSE average:\t\t", nrmse_average
    print "NRMSE anytime:\t\t", nrmse_anytime
    print "NRMSE last:\t\t", nrmse_last
    
    # calculate misclassifications for averaged output
    class_average = np.zeros((370,9))
    # search for the highest possible value and set it to 1
    ind = testout_average.argmax(1)
    class_average[range(370),ind] = 1
    misclass_average = abs(class_average-targ_average).sum() / 2.
    
    # calculate misclassifications for last output
    class_last = np.zeros((370,9))
    # search for the highest possible value and set it to 1
    ind = testout_last.argmax(1)
    class_last[range(370),ind] = 1
    misclass_last = abs(class_last-targ_average).sum() / 2.

    # calc misclassifications for anytime (all) outputs
    class_all = np.zeros(testout_all.shape)
    ind = testout_all.argmax(1)
    class_all[range(testout_all.shape[0]),ind] = 1
    misclass_all = abs(class_all-target_all).sum() / 2.

    #-----------------------------------
    # some additional calculations for time shifted versions
    misclass_m3 = abs(class_all[0:-3]-target_all[3:]).sum() / 2.
    misclass_m2 = abs(class_all[0:-2]-target_all[2:]).sum() / 2.
    misclass_m1 = abs(class_all[0:-1]-target_all[1:]).sum() / 2.
    misclass_0 = abs(class_all[0:]-target_all[0:]).sum() / 2.
    misclass_1 = abs(class_all[1:]-target_all[0:-1]).sum() / 2.
    misclass_2 = abs(class_all[2:]-target_all[0:-2]).sum() / 2.
    misclass_3 = abs(class_all[3:]-target_all[0:-3]).sum() / 2.
    print "shifted misclassifications (anytime):", misclass_m3, misclass_m2, misclass_m1,
    print misclass_0, misclass_1, misclass_2, misclass_3

    #-----------------------------------
    
    print "MISCLASSIFICATIONS average outputs:\t", misclass_average, \
          "\t(", 100 * misclass_average / float(370), "% )"
    print "MISCLASSIFICATIONS anytime outputs:\t", misclass_all, \
          "\t(", 100 * float(misclass_all) / testout_all.shape[0], "% )"
    print "MISCLASSIFICATIONS last outputs:\t", misclass_last, \
          "\t(", 100 * misclass_last / float(370), "% )"

    # do some plots if necessary ...
    if show:
        pylab.figure()
        pylab.plot(testout_average)
        pylab.title("average test outputs")
        
        pylab.figure()
        pylab.plot(testout_last)
        pylab.title("last test outputs")

        pylab.figure()
        pylab.plot(testout_all)
        pylab.title("anytime test outputs")      

        pylab.show()

def setup_multiple_reservoirs():
    """ multiple small reservoirs
    """
    array_size = 100
    # make prototype
    prototype = setup_ESN_jaeger_nr1()
    prototype.setSize(10)
#    prototype.setInitParam(au.ALPHA, 0.7)
#    prototype.setInitParam(au.CONNECTIVITY, 0.5)
    model = au.DoubleArrayESN( prototype, array_size )
    model.trainnoise = 1.e-5
    model.ds = 0
    # should we  choose randomly from the training exampless ?
    model.randrange = 1
    return model

def setup_ESN_Filter_DS():
    """ One delay&sum ESN with filters in the reservoir
    """
    model = filteresn.IIRESN()
    model.setSize(100)
    model.setInputs(14)
    model.setOutputs(9)
    model.setInitParam(au.ALPHA, 0.7)
    model.setInitParam(au.CONNECTIVITY, 0.2)
    model.setInitParam(au.IN_SCALE, 1.5)
    model.setInitParam(au.IN_CONNECTIVITY, 1.)
    model.setOutputAct(au.ACT_TANH)
    model.trainnoise = 1e-5
    # delay and sum ESN
    model.setSimAlgorithm(au.SIM_FILTER)
    model.setTrainAlgorithm( au.TRAIN_PI )
#    model.setSimAlgorithm(au.SIM_FILTER_DS)
    #model.setSimAlgorithm(au.SIM_SQUARE)
#    model.setTrainAlgorithm( au.TRAIN_DS_PI )
#    model.setInitParam(au.DS_USE_CROSSCORR)
    model.maxdelay = 7
#    model.setInitParam(au.DS_MAXDELAY, model.maxdelay)
#    model.setInitParam(au.DS_EM_ITERATIONS, 1)
#    model.setInitParam(au.EM_VERSION, 1)
    model.ds = 0
    # filter ESN parameters
#    model.setStdESN()
    model.setLinBPCutoffs(0.0001, 0.1, 8)
#    model.setConstBPCutoffs(f=[0.00001,0.01], bw=2)
    model.plot_filters()
    # should we  choose randomly from the training exampless ?
    model.randrange = 1
    return model

def setup_ESN_DS():
    """ One delay&sum ESN for the simulation
    """
    model = au.DoubleESN()
    model.setSize(100) # 200 is bissl besser
    model.setInputs(14)
    model.setOutputs(9)
    model.setInitParam(au.ALPHA, 0.7)
#    model.setInitParam(au.ALPHA, 0.2)
    model.setInitParam(au.CONNECTIVITY, 0.2)
    model.setInitParam(au.IN_SCALE, 1.5)
    model.setInitParam(au.IN_CONNECTIVITY, 1.)
    model.setOutputAct(au.ACT_TANH)
    model.trainnoise = 1e-5
    # delay and sum ESN
    model.setSimAlgorithm(au.SIM_FILTER_DS)
    #model.setSimAlgorithm(au.SIM_SQUARE)
    model.setTrainAlgorithm( au.TRAIN_DS_PI )
    model.setInitParam(au.DS_USE_CROSSCORR)
    model.maxdelay = 7
    model.setInitParam(au.DS_MAXDELAY, model.maxdelay)
    model.setInitParam(au.DS_EM_ITERATIONS, 1)
#    model.setInitParam(au.EM_VERSION, 1)
    model.ds = 1
    # leaky integrator neurons
#    leak_rate = 0.2
#    B = np.zeros((model.getSize(),2))
#    A = np.zeros((model.getSize(),2))
#    B[:,0] = 1.
#    A[:,0] = 1.
#    A[:,1] = (-1)*(1. - leak_rate)
#    model.setIIRCoeff(B,A)
    # should we  choose randomly from the training exampless ?
    model.randrange = 1
    return model

def setup_ESN_jaeger_nr1():
    """ The "Method Nr.1" of Jaegers paper.
    """
    model = au.DoubleESN()
    model.setSize(100)
    model.setInputs(14)
    model.setOutputs(9)
    #model.setInitParam(au.ALPHA, 0.2)
    model.setInitParam(au.ALPHA, 0.7)
    model.setInitParam(au.CONNECTIVITY, 0.2)
    model.setInitParam(au.IN_SCALE, 1.5)
    model.setInitParam(au.IN_CONNECTIVITY, 1.)
    model.setOutputAct(au.ACT_TANH)
    # leaky integrator ESN
    #model.setSimAlgorithm(au.SIM_LI)
    #model.setInitParam(au.LEAKING_RATE, 0.2)
    model.trainnoise = 1.e-5
    model.ds = 0
    # should we  choose randomly from the training exampless ?
    model.randrange = 1
    return model

def run_simulation():
    """ Performs simulation like "Method Nr.1" of Jaegers paper.
    """
    # select the network type
    model = setup_ESN_jaeger_nr1()
    #model = setup_ESN_DS()
#    model = setup_ESN_Filter_DS()

    
    #-------------------------------------------------------------------------- 
    print "Initialization ..."
    model.init()
    
    print "Load Benchmark Data ..."
    trainInputs, trainOutputs, testInputs, testOutputs, shifts = load_data()
    
    # shift trainOutputs into range [-0.9,0.9]
    for n in range(len(trainOutputs)):
        trainOutputs[n] = 1.8 * trainOutputs[n] - 0.9
    
    print "Training ..."
    train_esn_nr1(trainInputs, trainOutputs, model)
    
    print "Testing ..."
    out_anytime, out_average = test_esn_nr1(testInputs, model)
    
    print "Writing results to disk ..."
    
    # calculate averaged targets
    target_average = np.zeros((len(testInputs),9))
    for n in range( len(testInputs) ):
        target_average[n] = testOutputs[n][0]
    
    # write data to disk
    data = shelve.open("vowelresults-nr1.dat")
    data["testout_anytime"] = out_anytime
    data["testout_average"] = out_average
    data["target_anytime"] = testOutputs
    data["target_average"] = target_average
    data["test_looprange"] = model.test_looprange
    data.close()
    
    print "... finished !"


#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    run_simulation()
    plot_results()
