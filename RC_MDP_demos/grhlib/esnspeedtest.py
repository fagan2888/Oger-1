import aureservoir as au
import numpy as np
import time
from scipy import sparse

# TODO: versuchen das ganze mit cython: http://wiki.cython.org/tutorials/numpy,
#       schaut sehr uebersichtlich und effizient aus !

class ProfileESN(au.DoubleESN):
    """ Simple ESN for speed comparison.
    """
    
    def init(self):
        au.DoubleESN.init(self)
        
    def train(self, indata, outdata, washout):
        au.DoubleESN.train(self,indata,outdata,washout)
    
    def trainDummy(self):
        """ dummy for training, sets wout and other matrices
        """
        self.Wout = np.random.rand(self.getOutputs(), \
                              self.getSize()+self.getInputs()) * 2 - 1
        self.Wout = np.asfarray(self.Wout)
        self.setWout(self.Wout)
        
        # set other data
        self.W = np.zeros((self.getSize(), self.getSize()))
        self.getW(self.W)
        self.Win = self.getWin()
        self.Wback = self.getWback()
        self.x = np.zeros((self.getSize()))
        
        # for sparse simulation
        self.Wsp = sparse.csr_matrix(self.W)
        
    def simulate(self, indata, outdata):
        au.DoubleESN.simulate(self,indata,outdata)
    
    def simulateNumpy(self, indata):
        """ Standard Simulation without feedback, implemented in numpy.
        """
        simsteps = indata.shape[1]
        outdata = np.zeros(indata.shape)
        
        for n in range(simsteps):
            # calculate new internal state
            self.x = np.dot(self.W, self.x)
            self.x += np.dot(self.Win, indata[:, n])
            self.x = np.tanh(self.x)
            
            # calculate output
            outdata[:,n] = np.dot(self.Wout, np.r_[self.x,indata[:,n]] )
        
        return outdata
    
    def simulateNumpySparse(self, indata):
        """ Standard Simulation without feedback, implemented in numpy with sparse matrix.
        """
        simsteps = indata.shape[1]
        outdata = np.zeros(indata.shape)
        
        for n in range(simsteps):
            # calculate new internal state
            self.x = self.Wsp.dot(self.x)
            self.x += np.dot(self.Win, indata[:, n])
            self.x = np.tanh(self.x)
            
            # calculate output
            outdata[:,n] = np.dot(self.Wout, np.r_[self.x,indata[:,n]] )
        
        return outdata


#===============================================================================
# MAIN function
#===============================================================================

if __name__ == "__main__":
    net = ProfileESN()
    net.setSize(100)
    net.setInitParam( au.CONNECTIVITY, 0.1 )
    net.setInitParam( au.ALPHA, 0.8 )
    net.setInitParam( au.FB_CONNECTIVITY, 0. )
    net.setReservoirAct( au.ACT_TANH )
    net.setOutputAct( au.ACT_LINEAR )
    net.setSimAlgorithm( au.SIM_STD )
    
    print "init network ..."
    net.init()
    
    print "training network ..."
    net.trainDummy()
    
    simsize = 10000
    indata = np.asfarray(np.random.rand(net.getInputs(), simsize))*2-1
    outdata = np.zeros((net.getOutputs(), simsize))
    
    print "simulate aureservoir ..."
    t = time.clock()
    net.simulate(indata, outdata)
    aureservoir_time = time.clock() - t
    
    print "simulate numpy sparse version ..."
    t = time.clock()
    numpy_outdata = net.simulateNumpySparse(indata)
    numpysparse_time = time.clock() - t
    
    print "simulate numpy version ..."
    t = time.clock()
    numpy_outdata = net.simulateNumpy(indata)
    numpy_time = time.clock() - t
    
    print "\nCPU Time:"
    print "---------"
    print "aureservoir:\t", aureservoir_time
    print "numpy:\t", numpy_time
    print "numpy sparse:\t", numpysparse_time

