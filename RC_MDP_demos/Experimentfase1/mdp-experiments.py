#===============================================================================
# Some experiments with unsupervised algorithms of the MDP Toolkit
#===============================================================================

from numpy import *
import pylab
import testsignals
import mdp
#import aureservoir as au
import sys
sys.path.append("../grhlib")
#import filteresn
from Engine.reservoir_nodes import ReservoirNode
from resample_nodes import ResampleNode


#===============================================================================
# utility signals
#===============================================================================

def am_mix_of_sine(size,noiselevel):
    x = zeros((size,3))
    mix = zeros((size,1))
    x[:,0] = testsignals.sine(size,0.1,0.33)
    x[:,1] = testsignals.sine(size,0.01854,0.6)
    x[:,2] = testsignals.sine(size,0.06354,0.2)
    noise = (random.rand(size)*2.-1.) * noiselevel
    mix[:,0] = x[:,0] + x[:,1] + x[:,1] + noise
    
    # make some amplitude modulation
    am = testsignals.sine(size,0.0015,1.)
    mix[:,0] = am * mix[:,0]
    
#    pylab.plot( mix[:,0] )
#    pylab.show()
#    exit(0)
    
    return mix

def plot_signals(orig, analyse):
    pylab.figure()
    pylab.subplot(4,1,1)
    pylab.plot( orig )
    pylab.subplot(4,1,2)
    pylab.plot( analyse[:,0] )
    pylab.subplot(4,1,3)
    pylab.plot( analyse[:,1] )
    pylab.subplot(4,1,4)
    pylab.plot( analyse[:,2] )
   
def plot_all_signals(orig, analyse):
    size = analyse.shape[1]+1
    pylab.figure()
    pylab.subplot(size,1,1)
    pylab.plot( orig )
    for n in range(size-1):
        pylab.subplot(size,1,n+2)
        pylab.plot( analyse[:,n] )

def plot_architecture(flow, htmlfile="test.html"):
    """ generate HTML grafic of the architecture """
    file = open(htmlfile, "w")
    file.write('<html>\n<head>\n<title>HiNetHTML Test</title>\n</head>\n<body>\n')
    hinet_html = mdp.hinet.HiNetHTML(file)
    hinet_html.parse_flow(flow)
    file.write('</body>\n</html>')
    file.close()


#===============================================================================
# algorithm tests
#===============================================================================

def sfa_multiplesines():
    size = 1000
    mix = am_mix_of_sine(size,0.01)
    
    # Flow to perform SFA in the space of polynomials
    # TimeFramesNode: the time windows size
    flow = mdp.Flow([mdp.nodes.TimeFramesNode(10), \
                     mdp.nodes.PolynomialExpansionNode(3), \
                     mdp.nodes.SFANode(output_dim=3)])
    
    # train and execute flow to get the slow features
    flow.train(mix)
    slow = flow(mix)

    plot_signals(mix,slow)


def pca_multiplesines():
    size = 1000
    mix = am_mix_of_sine(size,0.01)
    
    # perform PCA
    flow = mdp.Flow([mdp.nodes.TimeFramesNode(10), \
                     mdp.nodes.PCANode(output_dim=3, svd=True)])
    flow.train(mix)
    pca = flow(mix)
    
    plot_signals(mix,pca)


def isfa_multiplesines():
    size = 1000
    mix = am_mix_of_sine(size,0.01)
    
    # perform ISFA
    flow = mdp.Flow([mdp.nodes.TimeFramesNode(10), \
                     mdp.nodes.PolynomialExpansionNode(1), \
                     mdp.nodes.ISFANode( output_dim=3 )])
#                     mdp.nodes.NIPALSNode( output_dim=3 )])
    flow.train(mix)
    ica = flow(mix)
    
    plot_signals(mix,ica)


def reservoir_multiplesines():
    size = 1000
    mix = am_mix_of_sine(size,0.00001)
    
    # normalize mix
#    mix = mix / mix.max()
    
    # reservoir prototype
    #prot = au.DoubleESN()
    #prot.setInitParam( au.CONNECTIVITY, 0.2 )
    #prot.setInitParam( au.ALPHA, 0.8 )
    #prot.setSize(100)
#    prot.setNoise( 1e-3 )
#    prot.setReservoirAct( au.ACT_LINEAR )
    
    # define a Flow with a reservoir and a SFA output layer
#    flow = mdp.Flow([ReservoirNode(1, 50, 'float64', params), \
#                     mdp.nodes.TimeFramesNode(5), \
#                    mdp.nodes.ISFANode(output_dim=3)])
#                    mdp.nodes.NIPALSNode(output_dim=3)])
    flow = mdp.Flow([ReservoirNode(1, 100, dtype='float64'), 
#                     mdp.nodes.TimeFramesNode(10), \
                     mdp.nodes.SFANode(output_dim=3)])
#    flow = mdp.Flow([ReservoirNode(1, 50, 'float64', params), \
#                     mdp.nodes.PCANode(output_dim=3, svd=True) ])
#    flow = mdp.Flow([ReservoirNode(1, 50, dtype='float64', params),])
    flow.train(mix)
    slow = flow(mix)
    
    plot_signals(mix,slow)

def hierarchical_multiplesines():
    size = 1000
    mix = am_mix_of_sine(size,0.00001)
#    mix = am_mix_of_sine(size,0.01)
    
    # normalize mix
#    mix = mix / mix.max()
    
    # reservoir prototype
    #prot = au.DoubleESN()
    #prot.setInitParam( au.CONNECTIVITY, 0.2 )
    #prot.setInitParam( au.ALPHA, 0.8 )
    #prot.setSize(100)
#    prot.setNoise( 1e-3 )
#    prot.setReservoirAct( au.ACT_LINEAR )
    
    # Hierarchical Network with Reservoirs and SFA nodes
    layer1 = mdp.Flow([ReservoirNode(1, 100, dtype='float64'), \
                       mdp.nodes.SFANode(output_dim=3), \
#                       mdp.nodes.PCANode(output_dim=5,svd=True), \
                       ResampleNode(3, 0.4, window="hamming") ])
    #prot.setSize(50)
    layer2 = mdp.Flow([ReservoirNode(3, 50, dtype='float64'), \
                       mdp.nodes.SFANode(output_dim=3), \
#                       mdp.nodes.PCANode(output_dim=5,svd=True), \
                       ResampleNode(3, 0.4, window="hamming") ])
    #prot.setSize(50)
    layer3 = mdp.Flow([ReservoirNode(3, 50, dtype='float64'), \
#                       mdp.nodes.PCANode(output_dim=3,svd=True) ])
                       mdp.nodes.SFANode(output_dim=3) ])

    layer1.train(mix)
    slow1 = layer1(mix)
    layer2.train(slow1)
    slow2 = layer2(slow1)
    layer3.train(slow2)
    slow3 = layer3(slow2)
    
    print slow1.shape, slow2.shape, slow3.shape
    
    plot_all_signals(mix,slow1)
    plot_all_signals(mix,slow2)
    plot_all_signals(mix,slow3)


def reservoir_multi():
    size = 1000
    mix = am_mix_of_sine(size,0.0001)
    
    # reservoir prototype
   # prot = au.DoubleESN()
   # prot.setInitParam( au.CONNECTIVITY, 0.3 )
   # prot.setInitParam( au.ALPHA, 0.8 )
   # prot.setSize(10)
    
    switchboard = mdp.hinet.Switchboard(input_dim=1, connections=[0,0,0,0,0])
    layer = mdp.hinet.Layer([ReservoirNode(1, 10, dtype='float64'), \
                             ReservoirNode(1, 10, dtype='float64'), \
                             ReservoirNode(1, 10, dtype='float64'), \
                             ReservoirNode(1, 10, dtype='float64'), \
                             ReservoirNode(1, 10, dtype='float64')])
    flow = mdp.Flow([switchboard, layer, mdp.nodes.SFANode(input_dim=50,output_dim=3)])
    print flow
    print layer
    flow.train(mix)
    slow = flow(mix)
    
    # generate HTML grafic of the architecture
#    plot_architecture(flow)
    
    plot_signals(mix,slow)


def reservoir_analysis():
    size = 1000
    mix = am_mix_of_sine(size,0.00001)
    
    # normalize mix
#    mix = mix / mix.max()
    
    # reservoir prototype
    #prot = filteresn.IIRESN()
    #prot.setInitParam( au.CONNECTIVITY, 0.2 )
    #prot.setInitParam( au.ALPHA, 0.8 )
    #prot.setSize(100)
#   # prot.setNoise( 1e-3 )
#   # prot.setReservoirAct( au.ACT_LINEAR )
    #prot.setSimAlgorithm( au.SIM_FILTER )
    #prot.setLogBPCutoffs(f_start=0.01, f_stop=0.4, bw=0.2, fs=1.)
#    prot.setIIRCoeff(prot.B,prot.A,prot.serial)
    
    # define a Flow with a reservoir and a SFA output layer
    flow = mdp.Flow([ReservoirNode(1, 100, dtype='float64'),])
    #prot.setIIRCoeff(prot.B,prot.A,prot.serial)
    flow.train(mix)
    slow = flow(mix)
    print slow.shape
    
    plot_signals(mix,slow)


#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
#    pca_multiplesines()
#    sfa_multiplesines()
#    isfa_multiplesines()
#    reservoir_multiplesines()
#    hierarchical_multiplesines()
#    reservoir_multi()
    reservoir_analysis()
    
    pylab.show()
