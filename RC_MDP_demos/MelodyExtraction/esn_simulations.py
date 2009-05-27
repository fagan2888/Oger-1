#===============================================================================
# Simulations with ESNs for MelodyExtraction
#===============================================================================

import aureservoir as au
import numpy as np
import mdp
import time
import sys
sys.path.append("../grhlib")
from reservoir_nodes import ReservoirNode, LinearReadoutNode, IdentityNode
from reservoir_nodes import SquareStatesNode
from svm_nodes import BinaryLinearSVMNode
#from resample_nodes import ResampleNode
from classification_experiment import simulation, analysis, construct_data



def construct_au_esn(size=100,conn=0.1,ins=257,outs=6,washout=0,datatype='float64'):
    """ Aureservoir ESN.
    """
    model = au.DoubleESN()
    model.setSize(size)
#    model.setSize(400) # last output geht hier besser, mean auch
#    model.setSize(1000) # bei last output 0 testerror
    model.setInputs(ins)
    model.setOutputs(outs)
    model.setInitParam(au.ALPHA, 0.2)
#    model.setInitParam(au.ALPHA, 0.5)
    model.setInitParam(au.CONNECTIVITY, conn)
    model.setInitParam(au.IN_SCALE, 1)
    model.setInitParam(au.IN_CONNECTIVITY, 0.3)
    model.setOutputAct(au.ACT_TANH)
    model.setSimAlgorithm(au.SIM_LI)
    model.setInitParam(au.LEAKING_RATE, 0.2)
#    model.setSimAlgorithm(au.SIM_FILTER_DS)
#    model.setSimAlgorithm(au.SIM_SQUARE)   # bringt manchmal doch was !!!
#    model.setTrainAlgorithm( au.TRAIN_DS_PI )
#    model.setInitParam(au.DS_USE_CROSSCORR)
#    model.maxdelay = 100
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
#    model.trainnoise = 1.e-5
    model.trainnoise = 0.
    model.testnoise = 0.
    model.randrange = 0
    model.washout = washout
    
    # init ESN
    model.init()
    
    return model

def construct_mdp_esn(size=100,conn=0.1,ins=257,outs=6,washout=0,
                      svm=0,svm_C=5,lr=0.2,in_scale=1.,noise=0.,
                      datatype='float64'):
    """ A simple standard ESN.
    """
    outputs = outs
    prot = au.DoubleESN()
    prot.setSize(size)
#    prot.setSize(400) # last output geht hier besser
#    prot.setSize(1000) # bei last output 0 testerror
    prot.setInputs(ins)
    prot.setOutputs(outputs)
    prot.setInitParam(au.ALPHA, lr)
#    prot.setInitParam(au.ALPHA, 0.5)
    prot.setInitParam(au.CONNECTIVITY, conn)
    prot.setInitParam(au.IN_SCALE, in_scale)
    prot.setInitParam(au.IN_CONNECTIVITY, 0.3)
#    prot.setOutputAct(au.ACT_TANH)
    # leaky integrator ESN
    prot.setSimAlgorithm(au.SIM_LI)
    prot.setInitParam(au.LEAKING_RATE, lr)
    
    ins = prot.getInputs()
    size = prot.getSize()
    outs = prot.getOutputs()
    reservoir = ReservoirNode(ins, size, dtype=datatype, prototype=prot)
    if svm==1:
        # using a L2-loss primal SVM with eps = 0.01
        # (faster if we have many timesteps)
        readout = BinaryLinearSVMNode((ins+size),outs,C=svm_C,solver_type=2,eps=0.01)
#        readout = BinaryLinearSVMNode(20,outs,C=svm_C,solver_type=2,eps=0.01)
    else:
        readout = LinearReadoutNode((ins+size), outs, ignore_ind=washout)
    
    # build hierarchical mdp network
    con_array = np.r_[range(ins), range(ins)]
    switchboard = mdp.hinet.Switchboard(ins, connections=con_array)
    res = mdp.hinet.Layer([reservoir, IdentityNode(ins, ins)])
    flow = mdp.Flow([switchboard, res,
#                     ResampleNode(ins+size, 0.5, window="hamming"),
#                     mdp.nodes.PCANode(output_dim=20,svd=True),
#                     SquareStatesNode(size+ins),
                     readout])
    model = mdp.hinet.FlowNode(flow)
    
    # additional properties
    model.type = 'ESN_mdp'
    model.print_W = svm+1 # print output weights after training
    model.dtype = datatype
#    model.label = 0.9
#    model.trainnoise = 1e-4
    model.trainnoise = noise
    model.testnoise = 0.
    model.randrange = 0
    model.output_act = "tanh" # tanh
#    model.bias = 0.01
#    model.length_feature = 0
#    model.scale_min = 0.
#    model.scale_max = 1.
    model.reset_state = 1 # clear state for each example
#    model.feature_key = "features_fft1024" # chroma features
    
    return model


#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    
    # choose example
#    file = "data/train11_STFT.dat"
#    train_range = [0,1000]
#    test_range = [1000,1950]
#    notes = 6
#    file = "data/train12_STFT.dat"
#    train_range = [2220,3300]
#    test_range = [1200,2220]
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
    file = ["data/train10_STFT.dat", "data/train11_STFT.dat",
            "data/train12_STFT.dat", "data/train13_STFT.dat"]
    train_range = [ [1520,2650], [0,1000], [2220,3300], [330,1300] ]
    test_range = [ [2900,3800], [1000,1950], [1200,2220], [1550,2550] ]
    notes = 19
#    file = ["data/train10_STFT.dat", "data/train11_STFT.dat","data/train12_STFT.dat"]
#    train_range = [ [1520,2650], [0,1000], [2220,3300] ]
#    test_range = [ [2900,3800], [1000,1950], [1200,2220] ]
#    notes = 19


    # multi-file examples
#    train_files = ["data/ATasteOfHoney_STFT.dat", "data/AgainKravitz_STFT.dat"]
#    test_files = ["data/WithALittleHelpFromMyFriends_STFT.dat", "data/M05train12_STFT.dat"]
#    notes = 13
    
    # choose model
#    mf_args=(1,1 ,257,notes, 0) # size, sparsity, ins, outs, washout
#    mf_args=(100,0.1 ,257,notes, 0) # size, sparsity, ins, outs, washout
#    mf_args=(1,1,257,notes,0, 1,0.1) # zus: svm, C
    mf_args=(1000,0.01,257,notes, 0, 1,0.01) # zus: svm, C
#    model_func=construct_au_esn
    model_func=construct_mdp_esn
    
    # data scaling and simulation properties
    label = 0.9
    bias = 0.1
    scale_min = 0
    scale_max = 1
    scale_prop = "noscale" # global, local, noscale
    
    data_func = construct_data.multiclass_anytime2
    df_args = (file,train_range,test_range,label,bias,scale_min, scale_max,scale_prop)
    datafile = "esn.dat"
    
    # simulation
    #time1 = time.time()
    #simulation.run_simulation(model_func, mf_args, data_func, df_args, datafile)
    #time2 = time.time()
    #print "SIMULATION TIME:",time2-time1
    
    # analysis
    analysis.analyze_data(datafile,show_any=0)
