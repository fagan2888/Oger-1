#===============================================================================
# Simulations with ESNs for EverydaySounds classification
#===============================================================================

import aureservoir as au
import numpy as np
import mdp
import sys
sys.path.append("../grhlib")
from reservoir_nodes import ReservoirNode, LinearReadoutNode, IdentityNode, StateCompressionNode
from svm_nodes import BinaryLinearSVMNode
from classification_experiment import simulation, analysis, construct_data


def construct_au_esn(size=100,conn=0.1,datatype='float64'):
    """ Aureservoir ESN.
    """
    model = au.DoubleESN()
    model.setSize(size)
#    model.setSize(400) # last output geht hier besser, mean auch
#    model.setSize(1000) # bei last output 0 testerror
    model.setInputs(15)
    model.setOutputs(10)
#    model.setInitParam(au.ALPHA, 0.2)
    model.setInitParam(au.ALPHA, 0.85)
    model.setInitParam(au.CONNECTIVITY, conn)
    model.setInitParam(au.IN_SCALE, 2)
    model.setInitParam(au.IN_CONNECTIVITY, 1.)
    model.setOutputAct(au.ACT_TANH)
#    model.setSimAlgorithm(au.SIM_LI)
#    model.setInitParam(au.LEAKING_RATE, 0.2)
#    model.setSimAlgorithm(au.SIM_FILTER_DS)
#    model.setSimAlgorithm(au.SIM_SQUARE)   # bringt manchmal doch was !!!
#    model.setTrainAlgorithm( au.TRAIN_DS_PI )
#    model.setInitParam(au.DS_USE_CROSSCORR)
#    model.maxdelay = 50
#    model.setInitParam(au.DS_MAXDELAY, model.maxdelay)
#    model.setInitParam(au.DS_EM_ITERATIONS, 1)
#    model.setInitParam(au.EM_VERSION, 1)
    
    # make multiple ESNs
    multiple_esns = 0
    if multiple_esns:
        array_size = 20
        print "ArrayESN: Using",array_size,"reservoir"
        model = au.DoubleArrayESN( model, array_size )
    
    # has d+s readout (1) or arrayesn (2) ?
    model.ds = 0
    
    # additional properties
    model.type = 'ESN_au'
    model.dtype = datatype
    model.trainnoise = 1.e-9
    model.testnoise = 0.
    model.randrange = 0
    
    # init ESN
    model.init()
    
    return model

def construct_mdp_esn(size=100,conn=0.1,svm=0,svm_C=5,datatype='float64'):
    """ A simple standard ESN.
    """
    outputs = 1
    prot = au.DoubleESN()
    prot.setSize(size)
#    prot.setSize(400) # last output geht hier besser
#    prot.setSize(1000) # bei last output 0 testerror
    prot.setInputs(15)
    prot.setOutputs(outputs)
#    prot.setInitParam(au.ALPHA, 0.2)
    prot.setInitParam(au.ALPHA, 0.85)
    prot.setInitParam(au.CONNECTIVITY, conn)
    prot.setInitParam(au.IN_SCALE, 2)
    prot.setInitParam(au.IN_CONNECTIVITY, 1.)
#    prot.setOutputAct(au.ACT_TANH)
    # leaky integrator ESN
#    prot.setSimAlgorithm(au.SIM_LI)
#    prot.setInitParam(au.LEAKING_RATE, 0.2)
    
    washout = 20 # for each example !
    ins = prot.getInputs()
    size = prot.getSize()
    outs = prot.getOutputs()
    reservoir = ReservoirNode(ins, size, dtype=datatype, prototype=prot)
    if svm==1:
        # using a L2-loss primal SVM with eps = 0.01
        # (faster if we have many timesteps)
        readout = BinaryLinearSVMNode(ins+size,outs,C=svm_C,solver_type=2,eps=0.01)
#        readout = BinaryLinearSVMNode(20,outs,C=svm_C,solver_type=2,eps=0.01)
    else:
        readout = LinearReadoutNode(ins+size, outs, ignore_ind=washout)
    
    # build hierarchical mdp network
    con_array = np.r_[range(ins), range(ins)]
    switchboard = mdp.hinet.Switchboard(ins, connections=con_array)
    res = mdp.hinet.Layer([reservoir, IdentityNode(ins, ins)])
    flow = mdp.Flow([switchboard, res,
#                     ResampleNode(ins+size, 0.5, window="hamming"),
#                     mdp.nodes.PCANode(output_dim=20,svd=True),
                     readout])
    model = mdp.hinet.FlowNode(flow)
    
    # additional properties
    model.type = 'ESN_mdp'
    model.print_W = svm+1 # print output weights after training
    model.dtype = datatype
    model.trainnoise = 1.e-9
    model.testnoise = 0.
    model.randrange = 1
    model.output_act = "tanh" # tanh
    model.reset_state = 1 # clear state for each example
    
    return model

def construct_mdp_statecompr(compr_points=5,size=10,conn=0.5,datatype='float64'):
    """ An ESN with state compression, as in the JapVowels Experiment by Jaeger.
    """
    outputs = 1
    prot = au.DoubleESN()
    prot.setSize(size)
#    prot.setSize(400) # last output geht hier besser
#    prot.setSize(1000) # bei last output 0 testerror
    prot.setInputs(15)
    prot.setOutputs(outputs)
    prot.setInitParam(au.ALPHA, 0.2)
#    prot.setInitParam(au.ALPHA, 0.85)
    prot.setInitParam(au.CONNECTIVITY, conn)
    prot.setInitParam(au.IN_SCALE, 2.)
    prot.setInitParam(au.IN_CONNECTIVITY, 1.)
#    prot.setOutputAct(au.ACT_TANH)
    # leaky integrator ESN
    prot.setSimAlgorithm(au.SIM_LI)
    prot.setInitParam(au.LEAKING_RATE, 0.2)
    
    ins = prot.getInputs()
    size = prot.getSize()
    outs = prot.getOutputs()
    
    # construct mdp ESN
    reservoir = ReservoirNode(ins, size, dtype=datatype, prototype=prot)
    compr = StateCompressionNode(ins+size, support_points=compr_points)
    readout = LinearReadoutNode(compr_points*(size+ins),
                                outs, ignore=0, use_pi=1)
    # TODO: SVM readout auch noch probiern !

    # build hierarchical mdp network
    res = mdp.hinet.SameInputLayer([reservoir, IdentityNode(ins, ins)])
    flow = mdp.Flow([res, compr, readout])
    model = mdp.hinet.FlowNode(flow)

    # additional properties
    model.type = 'ESN_statecompr'
    model.dtype = datatype
    model.trainnoise = 0.0001
    model.testnoise = 0.
    model.randrange = 1 
    model.output_act = "tanh" # tanh
    model.reset_state = 1 # clear state for each example
    model.print_W = 1
    
    return model


#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    
    mf_args=(100,0.1) # size, sparsity
#    mf_args=(100,0.05,1,2) # size, sparsity, svm, C
#    mf_args=(3, 4, 1.) # state compression ESN
    model_func=construct_mdp_esn
#    model_func=construct_au_esn
#    model_func=construct_mdp_statecompr

    # data scaling and properties
    label = 0.9
    bias = 1
    scale_min = 0
    scale_max = 1
    length_feature = 1
    scale_prop = "local"
    
    data_func = construct_data.binary_classification_1fold
#    data_func = construct_data.multiclass_classification_1fold
    df_args = ("data/drip_mfcc_fr100.dat",
               "data/flow_mfcc_fr100.dat",
#    df_args = (MFCC_FILES,
               3,1,label,bias,scale_min,scale_max,length_feature,scale_prop)
    
    datafile = "delme_esn.dat"
    
    # 1 fold simulation
    simulation.run_simulation(model_func, mf_args, data_func, df_args, datafile)

    # data analysis
    analysis.analyze_data(datafile,1)
#    analysis.analyze_data_mean(datafile,1)
