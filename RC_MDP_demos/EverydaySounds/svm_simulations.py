#===============================================================================
# Simulations with SVMs for EverydaySounds classification
#===============================================================================

import svm
import numpy as np
import mdp
import shelve
import sys
import pylab
sys.path.append("../grhlib")
from svm_nodes import BinarySVMNode, BinaryLinearSVMNode
from classification_experiment import simulation, analysis, construct_data


# mfcc files for multiclass simulation
MFCC_FILES = ["data/deformation_mfcc_fr100.dat", "data/explosion_mfcc_fr100.dat",
              "data/friction_mfcc_fr100.dat", "data/pour_mfcc_fr100.dat",
              "data/whoosh_mfcc_fr100.dat","data/drip_mfcc_fr100.dat",
              "data/flow_mfcc_fr100.dat", "data/impact_mfcc_fr100.dat",
              "data/rolling_mfcc_fr100.dat", "data/wind_mfcc_fr100.dat"]

# TODO: Grid search mit die Parameter machen !

def grid_search(datafile):
    """ Makes a grid search on parameters C and gamma of the SVM.
    """
    # Grid Search Tips:
    #
    # We found that trying exponentially growing sequences of C and gamma 
    # is a practical method to identify good parameters
    # (for example C = 2^-5,2^-3,...2^15 and gamma = 2^-15,2^-13,...2^3)
    #
    # For very large data sets, a feasible approach is to randomly choose a subset of the
    # data set, conduct grid-search on them, and then do a better-region-only grid-search
    # on the complete data set.
    
    # -> zuerst grid search min svm_mean, cross falidation
    # -> dann davon ausgehend mit gesamten daten, single fold

    # model
    model_func=construct_svm_mean

    # parameter space/grid:
    C = np.logspace(-5,15,11,base=2)
#    C = [1.,2.]
    gamma = np.logspace(-15,3,10,base=2)
#    gamma = [0.001,10.]
    
    # variables for error measures
    len_C = len(C)
    len_gamma = len(gamma)
    nrmse = np.zeros((len_C,len_gamma))
    misclass = np.zeros((len_C,len_gamma))
    
    # go through all the variables
    cursearch=1
    for i in range(len_C):
        for j in range(len_gamma):
            print "\n--------------- Grid-Search ",cursearch,"/",
            print len_C*len_gamma,"----------------"
            print "current C:", C[i]
            print "current gamma:", gamma[j]
            args = (C[i],gamma[j])
            run_simulation.run_multiclass_simulation_single(model_func,args,datafile)
            nrmse[i,j], misclass[i,j] = run_simulation.analyze_data_mean(datafile,0)
            cursearch += 1
    
    # save results
    data = shelve.open(datafile)
    data["nrmse"] = nrmse
    data["misclass"] = misclass
    data["C"] = C
    data["gamma"] = gamma
    data.close()
    
    print "\nGrid Search finished !"
    
def analyze_gs_data(datafile):
    """ Shows the results of the grid search.
    """
    data = shelve.open(datafile)
    nrmse = data["nrmse"]
    misclass = data["misclass"]
    C = data["C"]
    gamma = data["gamma"]
    data.close()
    
    print "NRMSE:"
    print nrmse
    print "Misclassifications:"
    print misclass
    
    pylab.figure()
    pylab.contourf(np.log2(gamma),np.log2(C),misclass)
    pylab.title("Misclassifications (GridSearch)")
    pylab.xlabel('log2( gamma )')
    pylab.ylabel('log2( C )')
    pylab.colorbar()
    
    pylab.figure()
    pylab.contourf(np.log2(gamma),np.log2(C),nrmse)
    pylab.title("NRMSE (GridSearch)")
    pylab.xlabel('log2( gamma )')
    pylab.ylabel('log2( C )')
    pylab.colorbar()
    
    pylab.show()

def construct_svm_mean(type="linear",C=2.,gamma=0.1,datatype='float64'):
    """ SVM mean model.
    """
    inputs = 14
    outputs = 1
    
    # Parameters
    svm_C = C
    svm_gamma = gamma
    
    # construct model
    if(type=='linear'):
        model = BinaryLinearSVMNode(inputs,outputs,C=svm_C)
    else:
        params = svm.svm_parameter(kernel_type=svm.RBF, C=svm_C, gamma=svm_gamma)
        model = BinarySVMNode(inputs,outputs,params)
    
    # additional properties
    model.dtype = datatype
    model.randrange = 0
    model.type = 'SVM_mean'
    
    return model

def construct_svm_anytime(type="linear",C=2.,gamma=0.1,datatype='float64'):
    """ SVM anytime model.
    """
    inputs = 14
    outputs = 1
    
    # Parameters
    svm_C = C
    svm_gamma = gamma
    
    # construct model
    if(type=='linear'):
        # using a L2-loss primal SVM with eps = 0.01
        # (faster if we have many timesteps)
        model = BinaryLinearSVMNode(inputs,outputs,C=svm_C,solver_type=2,eps=0.01)
    else:
        params = svm.svm_parameter(kernel_type=svm.RBF, C=svm_C, gamma=svm_gamma)
        model = BinarySVMNode(inputs,outputs,params)
    
    # additional properties
    model.dtype = datatype
    model.randrange = 0
    model.type = 'SVM_any'
    
    return model


#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    
#    grid_search("data/gs_svm.dat")
#    analyze_gs_data("data/gs_svm.dat")
    
    mf_args=("nonlinear",2.**15,2.**3) # C and gamma
#    mf_args=("linear",100) # C and gamma
    model_func=construct_svm_mean
#    model_func=construct_svm_anytime

    # data scaling and properties
    label = 1
    bias = 0
    scale_min = -1
    scale_max = 1
    length_feature = 1
    scale_prop = "local"
    
    data_func = construct_data.binary_classification_1fold
#    data_func = construct_data.multiclass_classification_1fold
    df_args = ("data/drip_mfcc_fr100.dat",
               "data/flow_mfcc_fr100.dat",
#    df_args = (MFCC_FILES,
               3,1,label,bias,scale_min,scale_max,length_feature,scale_prop)
    
    datafile = "delme_svm.dat"
    
    
    # 1 fold simulation
    simulation.run_simulation(model_func, mf_args, data_func, df_args, datafile)

    # n-fold simulation
#    simulation.run_simulation_multifold(model_func,mf_args,data_func,df_args,datafile,3)
    

    # data analysis
#    analysis.analyze_data(datafile,1)
    analysis.analyze_data_mean(datafile,1)
