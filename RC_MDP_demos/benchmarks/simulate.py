import esn_simulate
import esn_simulate1
import numpy as np
import time

#===============================================================================
# MAIN function
#===============================================================================

def simulate(f,indata):
    t = time.clock()
    data = f(indata)
    return time.clock() - t

if __name__ == "__main__":
    netsize = 100
    conn = 0.1
    simsize = 10000
    
    net1 = esn_simulate.ESNDummy(netsize,conn)
    indata = np.asfarray(np.random.rand(net1.ins, simsize),dtype=np.double)*2-1
    
    print "\nCPU Time:"
    print "---------"
    print "numpy:\t", simulate(net1.simulateNumpy, indata)
    print "scipy sparse:\t", simulate(net1.simulateNumpySparse, indata)
    print "aureservoir:\t", simulate(net1.simulateAureservoir, indata)
    print "pysparse:\t", simulate(net1.simulatePysparse, indata)
#    print "pyublas sparse:\t", simulate(net1.simulatePyublasSparse, indata)
#    print "weave.blitz:\t", simulate(net1.simulateBlitz, indata)
    print "weave.inline:\t", simulate(net1.simulateInline, indata)

#    print "CYTHON numpy:\t", simulate(esn_simulate1.simulateNumpy, indata)
#    print "CYTHON numpy sparse:\t", simulate(net2.simulateNumpySparse, indata)
    print "---------"