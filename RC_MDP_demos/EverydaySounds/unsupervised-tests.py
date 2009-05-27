#===============================================================================
# test properties of various structures when used with audio data
#===============================================================================

from numpy import *
import pylab
import aureservoir as au
import mdp
import sys
import shelve
sys.path.append("../grhlib")
from reservoir_nodes import ReservoirNode
from resample_nodes import ResampleNode
import datasets

#===============================================================================
# plotting utilities
#===============================================================================

def plot_all_signals(orig, analyse):
    size = analyse.shape[1]+1
    pylab.figure()
    pylab.subplot(size,1,1)
    pylab.plot( orig )
    for n in range(size-1):
        pylab.subplot(size,1,n+2)
        pylab.plot( analyse[:,n] )

def plot_all_signals_lable(orig, analyse, label):
    size = analyse.shape[1]+1
    pylab.figure()
    pylab.subplot(size,1,1)
    pylab.plot( orig )
    pylab.plot( label-0.5, 'r' )
    for n in range(size-1):
        pylab.subplot(size,1,n+2)
        pylab.plot( analyse[:,n] )

def plot_hr(file="hierarchical_reservoir.dat"):
    data = shelve.open(file)
    
    plot_all_signals_lable(data["signal"],data["slow1"],data["label"])
    plot_all_signals_lable(data["signal"],data["slow2"],data["label"])
    plot_all_signals_lable(data["signal"],data["slow3"],data["label"])
    plot_all_signals_lable(data["signal"],data["slow4"],data["label"])
    plot_all_signals_lable(data["signal"],data["slow5"],data["label"])
    plot_all_signals_lable(data["signal"],data["slow6"],data["label"])
    
    pylab.show()

def plot_sfa(file="hierarchical_sfa.dat"):
    data = shelve.open(file)
    
    plot_all_signals(data["signal"],data["slow1"])
    plot_all_signals(data["signal"],data["slow2"])
    plot_all_signals(data["signal"],data["slow3"])
    
    pylab.show()

#===============================================================================
# some algorithms
#===============================================================================

def hierarchical_reservoir(file="hierarchical_reservoir.dat"):
#    signal = datasets.read_single_soundfile()
    signal,label = datasets.get_two_class_dataset(5)
    
    # normalize signal
#    signal = signal / signal.max()
    
    # reservoir prototype
    prot = au.DoubleESN()
    prot.setInitParam( au.CONNECTIVITY, 0.2 )
    prot.setInitParam( au.ALPHA, 0.8 )
    prot.setSize(100)
#    prot.setNoise( 1e-3 )
#    prot.setReservoirAct( au.ACT_LINEAR )
    
    # Hierarchical Network with Reservoirs and SFA nodes
    layer1 = mdp.Flow([ReservoirNode(1, 100, 'float64', prot), \
                       mdp.nodes.SFANode(output_dim=3), \
#                       mdp.nodes.PCANode(output_dim=5,svd=True), \
                       ResampleNode(3, 0.3, window="hamming") ])
#    prot.setSize(50)
    layer2 = mdp.Flow([ReservoirNode(3, 100, 'float64', prot), \
                       mdp.nodes.SFANode(output_dim=3), \
#                       mdp.nodes.PCANode(output_dim=5,svd=True), \
                       ResampleNode(3, 0.3, window="hamming") ])
#    prot.setSize(50)
    layer3 = mdp.Flow([ReservoirNode(3, 100, 'float64', prot), \
#                       mdp.nodes.PCANode(output_dim=3,svd=True) ])
                       mdp.nodes.SFANode(output_dim=3),
                       ResampleNode(3, 0.3, window="hamming") ])
    layer4 = mdp.Flow([ReservoirNode(3, 100, 'float64', prot), \
#                       mdp.nodes.PCANode(output_dim=3,svd=True) ])
                       mdp.nodes.SFANode(output_dim=3),
                       ResampleNode(3, 0.3, window="hamming") ])
    layer5 = mdp.Flow([ReservoirNode(3, 100, 'float64', prot), \
#                       mdp.nodes.PCANode(output_dim=3,svd=True) ])
                       mdp.nodes.SFANode(output_dim=3),
                       ResampleNode(3, 0.3, window="hamming") ])
    layer6 = mdp.Flow([ReservoirNode(3, 100, 'float64', prot), \
#                       mdp.nodes.PCANode(output_dim=3,svd=True) ])
                       mdp.nodes.SFANode(output_dim=3),
                       ResampleNode(3, 0.3, window="hamming") ])
    
    # train and execute the layers
    layer1.train(signal)
    slow1 = layer1(signal)
    layer2.train(slow1)
    slow2 = layer2(slow1)
    layer3.train(slow2)
    slow3 = layer3(slow2)
    layer4.train(slow3)
    slow4 = layer4(slow3)
    layer5.train(slow4)
    slow5 = layer5(slow4)
    layer6.train(slow5)
    slow6 = layer6(slow5)
    
    print signal.shape, slow1.shape, slow2.shape, slow3.shape
    
    data = shelve.open(file)
    data["signal"] = signal
    data["label"] = label
    data["slow1"] = slow1
    data["slow2"] = slow2
    data["slow3"] = slow3
    data["slow4"] = slow4
    data["slow5"] = slow5
    data["slow6"] = slow6
    data["layers"] = 6
    data.close()

def hierarchical_sfa(file="hierarchical_sfa.dat"):
    signal = datasets.read_single_soundfile()
#    signal,label = datasets.get_two_class_dataset(5, class1, class2)
    
    # normalize signal
#    signal = signal / signal.max()

    # Hierarchical Network with SFA nodes
    layer1 = mdp.Flow([mdp.nodes.TimeFramesNode(10), \
                       mdp.nodes.PolynomialExpansionNode(2), \
                       mdp.nodes.SFANode(output_dim=3), \
                       ResampleNode(3, 0.3, window="hamming") ])
    layer2 = mdp.Flow([mdp.nodes.TimeFramesNode(10), \
                       mdp.nodes.PolynomialExpansionNode(2), \
                       mdp.nodes.SFANode(output_dim=3), \
                       ResampleNode(3, 0.3, window="hamming") ])
    layer3 = mdp.Flow([mdp.nodes.TimeFramesNode(10), \
                       mdp.nodes.PolynomialExpansionNode(2), \
                       mdp.nodes.SFANode(output_dim=3), \
                       ResampleNode(3, 0.3, window="hamming") ])
    
    # train and execute the layers
    layer1.train(signal)
    slow1 = layer1(signal)
    layer2.train(slow1)
    slow2 = layer2(slow1)
    layer3.train(slow2)
    slow3 = layer3(slow2)
    
    data = shelve.open(file)
    data["signal"] = signal
#    data["label"] = label
    data["slow1"] = slow1
    data["slow2"] = slow2
    data["slow3"] = slow3
    data["layers"] = 3
    data.close()


#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    # Simulations:
#    hierarchical_reservoir()
#    hierarchical_sfa()
    print "simulation finished !"
    
    # Plots:
#    plot_hr()
    plot_sfa()
    