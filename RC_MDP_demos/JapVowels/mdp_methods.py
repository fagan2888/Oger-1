#===============================================================================
# Experiments with MDP for the Japanese Vowels Task
#===============================================================================

import sys
import numpy as np
import aureservoir as au
import mdp
import random

sys.path.append("../grhlib")
from reservoir_nodes import ReservoirNode, LinearReadoutNode, IdentityNode
import single_reservoir_methods as srm

# NEUEN TEST MACHEN:
# - welcher training daten in mehreren train() Aufrufen uebergibt
#   -> schaun dass dann auch das Gleiche rauskommt
#   -> der ruft jetzt naemlich schon  vorher stop_training() auf


# VERSUCHE:
# - training methode so umstellen, dass immer ein washout:
#   -> geht jetzt mit der MDP Implementation
#
# - mehrere kleine reservoirs:
#   -> diese LinearReadoutNode gleich in reservoir_nodes dazu
#      - bei aureservoir test scripts code nehmen
#   -> delay+sum readout auch ?
#   -> dann kann man gleich mehr mit MDP herumspielen: z.B. SVM als readout oder
#      andere Nodes zusaetzlich probiern
#   -> auch am Eingang/Ausgang verschieden resamplen
#      -> kann trainingsdaten verlaengern -> reservoir kann einschwingen
#      -> nochmal die papers von den Belgiern anschaun

def train_model(train_in, train_out, model):
    # make a random or a linear range for the loop
    model.train_looprange = range(len(train_in))
    if model.randrange:
        random.shuffle(model.train_looprange)

    # train the network
    print "TODO: noise noch setzen fuer TRAINING !!!!!!!!!!!!!!!!!!!!!"
    # model.setNoise( model.trainnoise )
    for n in model.train_looprange:
        model.train([None, None, [(train_in[n], train_out[n])]])
    #model.setNoise( 0. )

def setup_single_ESN():
    """ The "Method Nr.1" of Jaegers paper, implemented with MDP.
    """
    res_size = 100
    prot = au.DoubleESN()
    prot.setSize(res_size)
    prot.setInputs(14)
    prot.setOutputs(9)
    #prot.setInitParam(au.ALPHA, 0.2)
    prot.setInitParam(au.ALPHA, 0.7)
    prot.setInitParam(au.CONNECTIVITY, 0.2)
    prot.setInitParam(au.IN_SCALE, 1.5)
    prot.setInitParam(au.IN_CONNECTIVITY, 1.)
    prot.setOutputAct(au.ACT_TANH)
    # leaky integrator ESN
    #prot.setSimAlgorithm(au.SIM_LI)
    #prot.setInitParam(au.LEAKING_RATE, 0.2)

    # construct ESN
    reservoir = ReservoirNode(input_dim=14, output_dim=res_size,
                              dtype='float64', prototype=prot)
    readout = LinearReadoutNode(input_dim=res_size+14, output_dim=9, use_pi=1)
    # build hierarchical mdp network
    switchboard = mdp.hinet.Switchboard(14,
                  connections=np.r_[range(14), range(14)])
    reslayer = mdp.hinet.Layer([reservoir, IdentityNode(14, 14)])
    model = mdp.Flow([switchboard, reslayer, readout])

    # other parameters
    model.trainnoise = 1.e-5
    model.randrange = 1
    return model

def run_simulation():
    """ Performs the simulation.
    """
    # select the model
    model = setup_single_ESN()
    
    #-------------------------------------------------------------------------- 
    print "Loading Benchmark Data ..."
    trainInputs,trainOutputs,testInputs,testOutputs,shifts = srm.load_data()
    
    # shift trainOutputs into range [-0.9,0.9]
    for n in range(len(trainOutputs)):
        trainOutputs[n] = 1.8 * trainOutputs[n] - 0.9
    
    print "Training ..."
    train_model(trainInputs, trainOutputs, model)
    
#    print "Testing ..."
#    out_anytime, out_average = test_esn_nr1(testInputs, model)
#    
#    print "Writing results to disk ..."
#    
#    # calculate averaged targets
#    target_average = np.zeros((len(testInputs),9))
#    for n in range( len(testInputs) ):
#        target_average[n] = testOutputs[n][0]
#    
#    # write data to disk
#    data = shelve.open("vowelresults-nr1.dat")
#    data["testout_anytime"] = out_anytime
#    data["testout_average"] = out_average
#    data["target_anytime"] = testOutputs
#    data["target_average"] = target_average
#    data["test_looprange"] = model.test_looprange
#    data.close()
#    
#    print "... finished !"

#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    run_simulation()
