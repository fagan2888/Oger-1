import numpy as np
from scipy import sparse

cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t


# global variables

cdef int N = 100
cdef ins = 1
cdef outs = 1

cdef np.ndarray Wout = np.asfarray(np.random.rand(outs, N+ins), dtype=DTYPE) * 2 - 1
cdef np.ndarray W = np.asfarray(np.random.rand(N, N), dtype=DTYPE) * 2. - 1.
cdef np.ndarray Win = np.asfarray(np.random.rand(N, ins), dtype=DTYPE) * 2 - 1
cdef np.ndarray x = np.zeros((N), dtype=DTYPE)

 
def simulateNumpy(np.ndarray indata):
    """ Standard Simulation without feedback, implemented in numpy.
    """
    cdef int simsteps = indata.shape[1]
    cdef np.ndarray outdata = np.zeros((indata.shape[0],indata.shape[1]), dtype=DTYPE)
    
    global x
    global W
    global Win
    global Wout
    
    cdef int n
    for n in range(simsteps):
        # calculate new internal state
        x = np.dot(W, x)
        x += np.dot(Win, indata[:, n])
        x = np.tanh(x)
        
        # calculate output
        outdata[:,n] = np.dot(Wout, np.r_[x,indata[:,n]] )
    
    return outdata


#    def simulateNumpySparse(self, indata):
#        """ Standard Simulation without feedback, implemented in numpy with sparse matrix.
#        """
#        simsteps = indata.shape[1]
#        outdata = np.zeros(indata.shape)
#        
#        for n in range(simsteps):
#            # calculate new internal state
#            self.x = self.Wsp.dot(self.x)
#            self.x += np.dot(self.Win, indata[:, n])
#            self.x = np.tanh(self.x)
#            
#            # calculate output
#            outdata[:,n] = np.dot(self.Wout, np.r_[self.x,indata[:,n]] )
#        
#        return outdata
