#===============================================================================
# Parameter Search for ESN simulation
#===============================================================================

import numpy as np
import shelve
import esn_simulations
import sys
import copy
sys.path.append("../grhlib")
from classification_experiment import simulation, analysis, construct_data


def do_simulation(model_args, data, datafile="esn.dat"):
    """ performs the actual simulation
    """
    # set the data
    if data==1:
        file = ["data/train10_STFT.dat", "data/train11_STFT.dat",
                "data/train12_STFT.dat", "data/train13_STFT.dat"]
        train_range = [ [1520,2650], [0,1000], [2220,3300], [330,1300] ]
        test_range = [ [2900,3800], [1000,1950], [1200,2220], [1550,2550] ]
    elif data==2:
        file = ["data/train10_STFT.dat", "data/train11_STFT.dat","data/train12_STFT.dat"]
        train_range = [ [1520,2650], [0,1000], [2220,3300] ]
        test_range = [ [2900,3800], [1000,1950], [1200,2220] ]
    else:
        print "ERROR - wrong data argument !"
        exit(-1)
    
    # data scaling and simulation properties
    label = 0.9
    bias = 0.1
    scale_min = 0
    scale_max = 1
    scale_prop = "noscale" # global, local, noscale
    
    mf_args=model_args
    model_func=esn_simulations.construct_mdp_esn
    data_func = construct_data.multiclass_anytime2
    df_args = (file,train_range,test_range,label,bias,scale_min, scale_max,scale_prop)
    
    # simulation
    simulation.run_simulation(model_func, mf_args, data_func, df_args, datafile)
    
    # analysis
    nrmse, misclass = analysis.analyze_data(datafile)
    
    return nrmse, misclass


def check_parameter(model_args, data, repetitions, result_file,
                    param_nr, param_values):
    """  param_nr       --    the parameter index in model_args to check
         param_values   --    list with values for this parameter
    """
    
    diff_params = len(param_values)
    nrmse = np.zeros((repetitions, diff_params))
    misclass = np.zeros((repetitions, diff_params))
    
    cursearch=1
    for i in range(repetitions):
        for j in range(diff_params): 
            print "\n--------------- Parameter-Search ",cursearch,"/",
            print diff_params*repetitions,"----------------"
            print "parameter:",result_file
            print "current repetition:",i+1,"of",repetitions
            print "current parameter value:",param_values[j]
            
            # set parameters
            args = copy.deepcopy(model_args)
            args[param_nr] = param_values[j]
            
            nrmse[i,j],misclass[i,j] = do_simulation(args, data)
            
            cursearch += 1

    # save results
    data = shelve.open(result_file)
    data["nrmse"] = nrmse
    data["misclass"] = misclass
    data["model_args"] = model_args
    data["param_nr"] = param_nr
    data["param_values"] = param_values
    data.close()
    
    print "\nCurrent Parameter Search finished !\n\n"

#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print "give 1 or 2 as argument for different processes !"
        exit(-1)
    
    process = int(sys.argv[1])
    if process!=1 and process!=2 and process!=3:
        print "wrong process number!"
        exit(-1)
    
    # model_args: standard arguments
    # data: 1=4erPack, 2=3erPack
    # repetitions: per parameter
    # result_file: where to store the results
    
    
    # default parameters
    model_args=[1000, 0.1, 257, 19, 0, 1, 0.001, 0.3, 0.3, 0.]
    data = 1
    repetitions = 6

    #------------------------------------------------------------------------------ 
    # now some simulations
    
    if process == 1:
        # size
        #result_file = "results/esn_ps3_size.dat"
        #param_nr = 0
        #param_values = [800, 1500]
        #check_parameter(model_args,data,repetitions,result_file,param_nr,param_values)
        
        # leaking rate
        #result_file = "results/esn_ps3_leakrate.dat"
        #param_nr = 7
        #param_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.9]
        #check_parameter(model_args,data,repetitions,result_file,param_nr,param_values)
        
        # input scale
        #result_file = "results/esn_ps3_inscale.dat"
        #param_nr = 8
        #param_values = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2]
        #check_parameter(model_args,data,repetitions,result_file,param_nr,param_values)

        # C
        result_file = "results/esn_ps4_C1.dat"
        param_nr = 6
        param_values = [1.e-7, 1.e-6, 0.1, 1, 10, 100]
        check_parameter(model_args,data,repetitions,result_file,param_nr,param_values)
        
    elif process == 2:
        # noise
        #result_file = "results/esn_ps3_noise.dat"
        #param_nr = 9
        #param_values = [1.e-3, 1.e-5, 1.e-7, 1.e-9, 1.e-11, 0.]
        #check_parameter(model_args,data,repetitions,result_file,param_nr,param_values)
        
        # C
        result_file = "results/esn_ps4_C2.dat"
        param_nr = 6
        param_values = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01]
        check_parameter(model_args,data,repetitions,result_file,param_nr,param_values)

    else:
        # repeat simulations with one parameter set
        
        # parameters
        model_args2=[1000, 0.1, 257, 19, 0, 1, 0.001, 0.3, 0.3, 0.]
        data2 = 1
        repetitions2 = 20

        for n in range(repetitions2):
            datafile2 = "results/esn_simulation_"+str(n)+".dat"
            do_simulation(model_args2, data2, datafile2)

