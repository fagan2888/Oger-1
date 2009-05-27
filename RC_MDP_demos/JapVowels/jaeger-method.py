#===============================================================================
# Python Implementation of Jaegers Method
# (see folder ./JaegerPaper/)
#===============================================================================

import shelve
import numpy as np
import aureservoir as au
from scipy.linalg import pinv
import pylab
import sys
sys.path.append("../grhlib")
import errormeasures


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

def collect_states_supp(trainInputs, net):
    """ Special state collection method of Jaegers JapVowel paper.
    """
    nrSamples = len(trainInputs)   # number of training examples
    net.size = net.getSize()
    
    # collect only suppNr states for one example
    stateSize = net.getInputs()+net.size
    stateCollection = np.zeros(( nrSamples, net.suppNr*stateSize ))
    
    for i in range(nrSamples):
        # time steps of current example
        l = trainInputs[i].shape[0]
        
        # calculate the whole state matrix for current example
        curin = trainInputs[i].T
        stateCollection_i = np.zeros((l,net.size))
        net.resetState() # reset state to zero
        net.collectStates(curin, stateCollection_i, 0)
        
        # now extract only suppNr states per example
        for s in range(net.suppNr-1): # the first two states
            ind = (s+1)*(l-1)/float(net.suppNr)
            lowerInd = np.floor( ind )
            
            # make an interpolation of states between indezes
            a = ind - lowerInd
            addState = (1-a) * stateCollection_i[lowerInd,:]
            addState += a * stateCollection_i[lowerInd+1,:]
            addInput = (1-a) * trainInputs[i][lowerInd,:]
            addInput += a * trainInputs[i][lowerInd+1,:]
            
            # finally collect the state and inputs
            ind2 = s*stateSize
            stateCollection[i,ind2:ind2+net.size] = addState
            stateCollection[i,ind2+net.size:ind2+stateSize] = addInput

        # the last state
        addState = stateCollection_i[-1,:]
        addInput = trainInputs[i][-1,:]
        # and append it to stateCollection matrix
        ind2 = (net.suppNr-1)*stateSize
        stateCollection[i,ind2:ind2+net.size] = addState
        stateCollection[i,ind2+net.size:ind2+stateSize] = addInput

#        # gleich wie oben, nur optimiert (ohne interpolation)
#        # -> PROBIERN ob das auch funktioniert
#        for s in range(net.suppNr):
#            ind = np.floor( (s+1)*(l-1)/float(net.suppNr) )
#            
#            # finally write data in stateCollection matrix
#            ind2 = s*stateSize
#            stateCollection[i,ind2:ind2+net.size] = stateCollection_i[ind,:]
#            stateCollection[i,ind2+net.size:ind2+stateSize] = trainInputs[i][ind,:]
    
    return stateCollection

def train_esn(trainInputs, trainOutputs, net):
    """ Special training method for the Japanese Vowel task.
    """
    nrSamples = len(trainInputs)   # number of training examples
    
    # collect one big state per example
    stateCollection = collect_states_supp(trainInputs, net)
    
    # collect one teacher signals per example
    teacherCollection = np.zeros((nrSamples, net.getOutputs()))
    for i in range(nrSamples):
        teacherCollection[i,:] = trainOutputs[i][0,:]
    # check and undo output activation function
    if net.getOutputAct() == au.ACT_TANH:
        teacherCollection = np.arctanh( teacherCollection )
        
    # finally compute output weights
    # calc pseudo inverse: wout = pinv(states) * target
    net.wout = ( np.dot(pinv(stateCollection), teacherCollection) )

def test_esn(indata, net):
    """ Special simulation method for the Japanese Vowel task.
    """
    
    # collect one big state per example
    stateCollection = collect_states_supp([indata,], net).flatten()
    
    # calculate one ouput vector for this example
    outputSequence = np.dot( stateCollection, net.wout )
    
    # check and apply output activation function
    if net.getOutputAct() == au.ACT_TANH:
        outputSequence = np.tanh( outputSequence )
    
    return outputSequence


def plot_jager_results(show_all=0):
    """ Plot and calculate results as in Jeagers paper.
    """
    # first load the data
    data = shelve.open("vowelresults.dat")
    trainOutsIndividual = data["trainOutsIndividual"]
    testOutsIndividual = data["testOutsIndividual"]
    trainTargetsIndividual = data["trainTargetsIndividual"]
    testTargetsIndividual = data["testTargetsIndividual"]
    nExperts = data["nExperts"]
    shifts = data["shifts"]
    data.close()
    
    # the combined voter sets (must be divideable by nExperts !)
    voteSets = [1, 5, 10, 25, 50, 100, 250, 500]  # for nExperts = 500
    #voteSets = [1, 2]  # for nExperts = 2
    #voteSets = [1, 2, 4]  # for nExperts = 4
    #voteSets = [1, 2, 4, 8]  # for nExperts = 8
    #voteSets = [1, 2, 4, 8, 16]  # for nExperts = 16
    nVoteSets = len(voteSets);
    
    # allocate mem for the diagnostic results
    meanResultsNRMSE_train = []
    meanResultsClass_train = []
    meanResultsNRMSE_test = []
    meanResultsClass_test = []
    
    # go through all the combined voting sets
    for voteSet in voteSets:
        nrSets = nExperts/voteSet
	
        curTrainClass = np.zeros(nrSets)
        curTestClass = np.zeros(nrSets)
        curTrainNRMSE = np.zeros(nrSets)
        curTestNRMSE = np.zeros(nrSets)
        
        # average over experts
        for l in range(nrSets):
            lumpedTrainOuts = trainOutsIndividual[:,:,l*voteSet:(l+1)*voteSet].mean(2)
            lumpedTestOuts = testOutsIndividual[:,:,l*voteSet:(l+1)*voteSet].mean(2)
            
            trainClassifications = np.zeros((270,9))
            testClassifications = np.zeros((370,9))
            # search for the highest possible value and set it to 1
            ind = lumpedTrainOuts.argmax(1)
            trainClassifications[range(270),ind] = 1
            ind = lumpedTestOuts.argmax(1)
            testClassifications[range(370),ind] = 1
            
            # calculate NRMSE for current set of experts
            curTrainNRMSE[l] = errormeasures.nrmse(lumpedTrainOuts.flatten(),
                                                   trainTargetsIndividual.flatten())
            curTestNRMSE[l] = errormeasures.nrmse(lumpedTestOuts.flatten(),
                                                   testTargetsIndividual.flatten())

            # calculate nr of miscalculations
            trainMisclassifications = abs(trainClassifications - trainTargetsIndividual)
            testMisclassifications = abs(testClassifications - testTargetsIndividual)
            curTrainClass[l] = trainMisclassifications.sum() / 2.
            curTestClass[l] = testMisclassifications.sum() / 2.
            
            # plot results for the first voters of a given set
            if l==0 and show_all:
                pylab.figure()
                pylab.plot(lumpedTrainOuts)
                pylab.title("train outs, "+str(voteSet)+" voters")
                
                pylab.figure()
                pylab.plot(lumpedTestOuts)
                pylab.title("test outs, "+str(voteSet)+" voters")
            
        # mean NRMSE outputs berechnen (einfach an die Liste appenden):
        meanResultsClass_train.append( curTrainClass.mean() )
        meanResultsClass_test.append( curTestClass.mean() )
        meanResultsNRMSE_train.append( curTrainNRMSE.mean() )
        meanResultsNRMSE_test.append( curTestNRMSE.mean() )
    
    print "train NRMSEs:", meanResultsNRMSE_train
    print "test NRMSEs:", meanResultsNRMSE_test
    print "train misclassifications:", meanResultsClass_train
    print "test misclassifications:", meanResultsClass_test
    
    pylab.figure()
    pylab.plot(voteSets, meanResultsClass_train, 'b')
    pylab.plot(voteSets, meanResultsClass_test, 'r')
    pylab.title('Train (b) and Test (r) Misclassification vs. vote set size')

    pylab.figure()
    pylab.plot(voteSets, meanResultsNRMSE_train, 'b')
    pylab.plot(voteSets, meanResultsNRMSE_test, 'r')
    pylab.title('Train (b) and Test (r) NRMSE vs. vote set size')

    pylab.show()


def run_jaeger_simulation():
    """ Creates and initializes all networks.
    """
    nExperts = 500     # Number of voting experts
    #nExperts = 16      # for testing
    nInternalUnits = 4 # reservoir size per voter
    nInputUnits = 14   # number of input units
    nOutputUnits = 9   # number of output units, fixed at 9 for this task
    suppNr = 3         # called D in Jaegers paper (support points),
                       # D semented reservoir states per sample

    # make a reservoir model
    model = au.DoubleESN()
    model.setSize(nInternalUnits)
    model.setInputs(nInputUnits)
    model.setOutputs(nOutputUnits)
    model.setInitParam(au.ALPHA, 0.2)
    model.setInitParam(au.CONNECTIVITY, 1.)
    model.setInitParam(au.IN_SCALE, 1.5)
    model.setInitParam(au.IN_CONNECTIVITY, 1.)
    model.setOutputAct(au.ACT_TANH)
    # leaky integrator ESN
    model.setSimAlgorithm(au.SIM_LI)
    model.setInitParam(au.LEAKING_RATE, 0.2)
    model.post()
    
    
    # Initialization

    trainOutsIndividual = np.zeros((270, nOutputUnits, nExperts))
    testOutsIndividual = np.zeros((370, nOutputUnits, nExperts))

    experts = []   # list with all reservoirs
    print "Initialization ..."
    for n in range(nExperts):
        newReservoir = au.DoubleESN(model)
        newReservoir.suppNr = suppNr
        experts.append(newReservoir)
        experts[n].init()   # init networks
    
    print "Load Benchmark Data ..."
    trainInputs, trainOutputs, testInputs, testOutputs, shifts = load_data()
    
    # shift trainOutputs into range [-0.9,0.9]
    for n in range(len(trainOutputs)):
        trainOutputs[n] = 1.8 * trainOutputs[n] - 0.9
    
    # Training
    print "Training ..."
    for n in range(nExperts):
        train_esn(trainInputs, trainOutputs, experts[n])
    
    # Testing
    print "Testing ..."
    
    # calculate output for each training example
    for n in range(nExperts):
        for i in range(270):
            trainOutsIndividual[i,:,n] = test_esn(trainInputs[i], experts[n])
    
        # now calc output for test examples
        for i in range(370):
            testOutsIndividual[i,:,n] = test_esn(testInputs[i], experts[n])
    
    # rescale signals into original scale
    trainOutsIndividual = (trainOutsIndividual+0.9) / 1.8
    testOutsIndividual = (testOutsIndividual+0.9) / 1.8
    
    # save data for analysis
    print "Writing results to disk ..."
    
    # get the rights targets
    trainTargetsIndividual = np.zeros((270, nOutputUnits))
    testTargetsIndividual = np.zeros((370, nOutputUnits))
    for i in range(270):
        trainTargetsIndividual[i,:] = trainOutputs[i][0,:]
    for i in range(370):
        testTargetsIndividual[i,:] = testOutputs[i][0,:]
    # rescale training targets
    trainTargetsIndividual = (trainTargetsIndividual+0.9) / 1.8
    
    # write data to disk
    data = shelve.open("vowelresults.dat")
    data["trainOutsIndividual"] = trainOutsIndividual
    data["testOutsIndividual"] = testOutsIndividual
    data["trainTargetsIndividual"] = trainTargetsIndividual
    data["testTargetsIndividual"] = testTargetsIndividual
    data["nExperts"] = nExperts
    data["shifts"] = shifts
    data.close()
    
    print "... finished !"



#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    #run_jaeger_simulation()
    plot_jager_results()
