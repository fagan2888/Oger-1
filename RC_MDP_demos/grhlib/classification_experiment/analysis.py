#===============================================================================
# General analysis methods for a classification experiment
#===============================================================================

import numpy as np
import shelve
import pylab
import sys
sys.path.append("../grhlib")
import errormeasures

def calc_misclassifications(data,target,label=0.9,datatype='float64'):
    """ Calculates misclassifications between the data and a target.
    """
    # first scale data from [-label,label] to [0,1]
    data1 = (data+label) / (2*label)
    targ1 = (target+label) / (2*label)
    
    classes = np.zeros(data.shape, dtype=datatype)
    ind = data.argmax(1)
    classes[range(data.shape[0]),ind] = 1
    
    misclassifications = abs(classes-targ1).sum() / 2.
    return misclassifications

def calc_misclassifications_binary(data,target,label=0.9,datatype='float64'):
    """ Calculates misclassifications between the data and a target for a
    binary classification task.
    """
    # first scale data from [-label,label] to [0,1]
    data1 = (data+label) / (2*label)
    targ1 = (target+label) / (2*label)
    
    # threshold is now 0.5
    data1[np.nonzero(data1<0.5)] = 0
    data1[np.nonzero(data1>=0.5)] = 1

    misclassifications = abs(data1-targ1).sum()
    return misclassifications


#===============================================================================
# high level interface
#===============================================================================

def analyze_data_mean(datafile="exp-a4-test.dat", show_plots=0):
    """ Analysis of the simulation results (for SVM_mean).
    """
    # read data
    data = shelve.open(datafile)
    testout = data["testout"]
    target = data["target"]
    test_looprange = data["test_looprange"]
    datatype = data["datatype"]
    data.close()
    
    # make target
    examples = len(test_looprange)
    outputs = testout[0].shape[1]
    test = np.zeros((examples,outputs),dtype=datatype)
    targ = np.zeros((examples,outputs),dtype=datatype)
    for n in range(examples):
        test[n] = testout[n]
        targ[n] = target[test_looprange[n]][0]

    # calculate NRMSE
    nrmse = errormeasures.nrmse(test, targ)
    print "NRMSE:\t\t", nrmse

    # calculate misclassifications
    label = abs(target[0][0,0])
    if outputs==1:
        # binary classification
        misclass = calc_misclassifications_binary(test,targ,label,datatype)
    else:
        # multiclass
        misclass = calc_misclassifications(test,targ,label,datatype)

    print "MISCLASSIFICATIONS:\t", misclass, \
          "\t(", 100 * misclass / float(examples), "% )"

    # make some plots
    if show_plots:
        
        if outputs==1:
            # binary
            pylab.figure()
            pylab.plot(test)
            pylab.plot(targ,'r')
            pylab.title("testout and target (red)")
        else:
            # multiclass
            pylab.figure()
            pylab.plot(test)
            pylab.plot(targ) # TODO: schaun dass die Farben gleich sind
            pylab.title("test and target outputs")
            
            exclass = 5
            pylab.figure()
            pylab.plot(test[:,exclass])
            pylab.plot(targ[:,exclass])
            pylab.title("test and target mean, class"+str(exclass))

        pylab.show()
        
    return nrmse, misclass

def analyze_data(datafile="exp-a4-test.dat", show_plots=0, show_any=0):
    """ Analysis of the simulation results.
    """
    # read data
    data = shelve.open(datafile)
    testout = data["testout"]
    target = data["target"]
    test_looprange = data["test_looprange"]
    datatype = data["datatype"]
    data.close()

    # calculate last and mean for each example
    outputs = target[0].shape[1]
    examples = len(test_looprange)
    testout_mean = np.zeros((examples,outputs),dtype=datatype)
    testout_last = np.zeros((examples,outputs),dtype=datatype)
    target_mean = np.zeros((examples,outputs),dtype=datatype)
    
    for n in range(examples):
#        discard = 200
#        if discard > testout[n].shape[0]:
#            discard = testout[n].shape[0]-5
        testout_mean[n] = testout[n].mean(0)
        testout_last[n] = testout[n][-1]
        target_mean[n] = target[test_looprange[n]][0]

    # make anytime outputs
    target_any = np.zeros((0,outputs),dtype=datatype)
    testout_any = np.zeros((0,outputs),dtype=datatype)
    for n in range(examples):
        testout_any = np.r_[testout_any, testout[n]]
        target_any = np.r_[target_any, target[test_looprange[n]]]
    
    print target_any
    
    # calculate NRMSE
    nrmse_anytime = errormeasures.nrmse(testout_any, target_any)
    nrmse_average = errormeasures.nrmse(testout_mean, target_mean)
    nrmse_last = errormeasures.nrmse(testout_last, target_mean)
    
    print "NRMSE mean:\t\t", nrmse_average
    print "NRMSE anytime:\t\t", nrmse_anytime
    print "NRMSE last:\t\t", nrmse_last

    # calculate misclassifications
    label = abs(target[0][0,0])
    if outputs==1:
        # binary classification
        misclass_any = calc_misclassifications_binary(testout_any,target_any,
                                                  label,datatype)
        misclass_mean = calc_misclassifications_binary(testout_mean,target_mean,
                                                   label,datatype)
        misclass_last = calc_misclassifications_binary(testout_last,target_mean,
                                                   label,datatype)
    else:
        # multiclass classification
        misclass_any = calc_misclassifications(testout_any,target_any,
                                               label,datatype)
        misclass_mean = calc_misclassifications(testout_mean,target_mean,
                                                label,datatype)
        misclass_last = calc_misclassifications(testout_last,target_mean,
                                                label,datatype)
    
    print "MISCLASSIFICATIONS mean outputs:\t", misclass_mean, \
          "\t(", 100 * misclass_mean / float(examples), "% )"
    print "MISCLASSIFICATIONS anytime outputs:\t", misclass_any, \
          "\t(", 100 * float(misclass_any) / testout_any.shape[0], "% )"
    print "MISCLASSIFICATIONS last outputs:\t", misclass_last, \
          "\t(", 100 * misclass_last / float(examples), "% )"

    # make some plots
    if show_plots:
        
        if outputs == 1:
            # binary classification
            pylab.figure()
            pylab.plot(testout_mean)
            pylab.plot(target_mean,'r')
            pylab.title("mean outputs (target=red)")

            pylab.figure()
            pylab.plot(testout_any)
            pylab.plot(target_any,'r')
            pylab.title("anytime outputs (target=red)")
        else:
            # multiclass
            pylab.figure()
            pylab.plot(testout_mean)
            pylab.plot(target_mean) # TODO: schaun dass die Farben gleich sind
            pylab.title("test and target mean outputs")
            
            exclass = 5
            pylab.figure()
            pylab.plot(testout_mean[:,exclass])
            pylab.plot(target_mean[:,exclass])
            pylab.title("test and target mean, class"+str(exclass))

    # show anytime plots
    if show_any:
        pylab.figure()
        pylab.subplot(211)
        pylab.plot(testout_any)
        pylab.title("test output")
        pylab.subplot(212)
        pylab.plot(target_any)
        pylab.title("target output")
        
    
    if show_any or show_plots:        
        pylab.show()
    
    return nrmse_anytime, misclass_any
