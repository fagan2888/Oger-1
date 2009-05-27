#===============================================================================
# Some Reservoir-SFA experiments with Wendelin
#===============================================================================

from numpy import *
import aureservoir as au
import pylab
import testsignals
import mdp
import sys
sys.path.append("../grhlib")
from reservoir_nodes import ReservoirNode


#===============================================================================
# helper functions
#===============================================================================

def plot_signals(orig, analyse):
    leng = analyse.shape[1] + 1

    pylab.figure()
    pylab.subplot(leng,1,1)
    pylab.plot( orig )
    
    for i in range(leng-1):
        pylab.subplot(leng,1,i+2)
        pylab.plot(analyse[:,i])

def phase_plot(mod, sfaout):
    pylab.figure()
    
#    print mod[9:].shape
#    print sfaout.shape
    pylab.plot(mod,sfaout[:,0])
   


#===============================================================================
# algorithm tests
#===============================================================================

def sfa_standard(testsignal,order=2,time=50):
    
    testsignal.shape = -1,1
    
    # Flow to perform SFA in the space of polynomials
    # TimeFramesNode: the time windows size
    sfa = mdp.Flow([mdp.nodes.TimeFramesNode(50), \
#                    mdp.nodes.PCANode(output_dim=5,svd=True), \
                    mdp.nodes.PolynomialExpansionNode(order), \
                    mdp.nodes.SFANode(output_dim=3)])
    
    # train and execute flow to get the slow features
    sfa.train(testsignal)
    slow = sfa(testsignal)

    #plot_signals(testsignal,slow)
    return slow

def reservoir_sfa(testsignal,size=100,conn=0.1):
    
    testsignal.shape = -1,1
    
    # reservoir prototype
    prot = au.DoubleESN()
    prot.setInitParam( au.CONNECTIVITY, conn )
    prot.setInitParam( au.ALPHA, 0.9 )
    prot.setSize(size)
    
    # Reservoir Node with SFA
    res_sfa = mdp.Flow([ReservoirNode(1, size, prototype=prot),
                        mdp.nodes.PCANode(output_dim=300,svd=True),
#                        mdp.nodes.PolynomialExpansionNode(2), \
                        mdp.nodes.SFANode(output_dim=3)])
    
    # train and execute flow to get the slow features
    res_sfa.train(testsignal)
    slow = res_sfa(testsignal)

    return slow


#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    
    # generate signal
    size = 10000
    time = 50
#    signal, mod = testsignals.fm_sine(size, 0.005, 0.001, 1, 2.5)
#    signal, mod = testsignals.randwalk_sine(size, 0.01, 1., 0.5, 5)
#    signal, mod = testsignals.am_sine(10000, 0.01, 0.0005)
    signal, mod = testsignals.fm_sine_special(size, 0.01, 1., 3)
    
    # add some noise
    testsignals.add_noise( signal, 1e-5 )
    
    pylab.plot(signal)
    pylab.figure()
    pylab.plot(mod)
    pylab.show()
    exit(0)
#    
    # the algorithms
#    sfa_out = sfa_standard(signal,2,time)
#    pylab.plot(signal[(time-1):],sfa_out[:,0])
    
    sfa_out = reservoir_sfa(signal,1000,0.05)
    pylab.plot(signal,sfa_out[:,0])

    plot_signals(signal,sfa_out)
  
    
    pylab.show()

