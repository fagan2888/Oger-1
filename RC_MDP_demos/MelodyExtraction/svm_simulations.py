#===============================================================================
# Simulations with SVMs for MelodyExtraction
#===============================================================================

import numpy as np
import mdp
import shelve
import svm
import sys
import pylab
sys.path.append("../grhlib")
from svm_nodes import BinarySVMNode, BinaryLinearSVMNode
from classification_experiment import simulation, analysis, construct_data


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
    
    model_func=construct_svm_anytime
    
    # data scaling and simulation properties
#    file = "data/train11_STFT.dat"
#    train_range = [0,1000]
#    test_range = [1000,1950]
#    notes = 6
#    file = "data/train12_STFT.dat"
#    train_range = [2220,3300]
#    test_range = [0,2220]
#    notes = 7
#    file = "data/train10_STFT.dat"
#    train_range = [1520,2650]
#    test_range = [2900,3800]
#    notes = 9
#    file = "data/train13_STFT.dat"
#    train_range = [330,1300]
#    test_range = [1550,2550]
#    notes = 8
    # multiple files
#    file = ["data/train10_STFT.dat", "data/train11_STFT.dat",
#            "data/train12_STFT.dat", "data/train13_STFT.dat"]
#    train_range = [ [1520,2650], [0,1000], [2220,3300], [330,1300] ]
#    test_range = [ [2900,3800], [1000,1950], [1200,2220], [1550,2550] ]
#    notes = 19
    file = ["data/train10_STFT.dat", "data/train11_STFT.dat","data/train12_STFT.dat"]
    train_range = [ [1520,2650], [0,1000], [2220,3300] ]
    test_range = [ [2900,3800], [1000,1950], [1200,2220] ]
    notes = 19
    
    label = 1
    bias = 0.1
    scale_min = -1
    scale_max = 1
    scale_prop = "noscale" # global, local, noscale
    
    data_func = construct_data.multiclass_anytime2
    df_args = (file,train_range,test_range,label,bias,scale_min, scale_max,scale_prop)
    
    
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

            mf_args=("nonlinear",C[i],gamma[j], 257,notes)
            
            # simulation
            simulation.run_simulation(model_func, mf_args, data_func, df_args, datafile)
            
            nrmse[i,j], misclass[i,j] = analysis.analyze_data(datafile)
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
    
    ind_min = np.where(misclass == misclass.min())
    print "Min.Misclass:",misclass[ind_min]
    print "Min.Misclass index:", ind_min
    print "C min:",C[ind_min[0]]
    print "gamma min:",gamma[ind_min[1]]
    
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


def construct_svm_anytime(type="linear",C=2.,gamma=0.1,ins=257,outs=6,datatype='float64'):
    """ SVM anytime model.
    """
    inputs = ins
    outputs = outs
    
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
#    analyze_gs_data("results/gs_svm_4erPack.dat")
#    exit(0)
    
    # choose example
#    file = "data/train11_STFT.dat"
#    train_range = [0,1000]
#    test_range = [1000,1950]
#    notes = 6
#    file = "data/train10_STFT.dat"
#    train_range = [1520,2650]
#    test_range = [2900,3800]
#    notes = 9
#    file = "data/train13_STFT.dat"
#    train_range = [330,1300]
#    test_range = [1550,2550]
#    notes = 8

    # multi-file examples
    file = ["data/train10_STFT.dat", "data/train11_STFT.dat",
            "data/train12_STFT.dat", "data/train13_STFT.dat"]
    train_range = [ [1520,2650], [0,1000], [2220,3300], [330,1300] ]
    test_range = [ [2900,3800], [1000,1950], [1200,2220], [1550,2550] ]
    notes = 19
#    file = ["data/train10_STFT.dat", "data/train11_STFT.dat","data/train12_STFT.dat"]
#    train_range = [ [1520,2650], [0,1000], [2220,3300] ]
#    test_range = [ [2900,3800], [1000,1950], [1200,2220] ]
#    notes = 19
    
    # choose model
    mf_args=("nonlinear",0.5,0.125, 257,notes) # C and gamma from Poliner paper
    model_func=construct_svm_anytime
    
    # data scaling and simulation properties
    label = 1
    bias = 0.1
    scale_min = -1
    scale_max = 1
    scale_prop = "noscale" # global, local, noscale
    
    data_func = construct_data.multiclass_anytime2
    df_args = (file,train_range,test_range,label,bias,scale_min, scale_max,scale_prop)
    datafile = "svm.dat"
    
    # simulation
    simulation.run_simulation(model_func, mf_args, data_func, df_args, datafile)
    
    # analysis
    analysis.analyze_data(datafile,show_any=1)
