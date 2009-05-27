import numpy as np
from scipy import sparse, weave
from scipy.weave import converters
import aureservoir as au
from pysparse import spmatrix
import pyublas

# TODO: 
#
# - pyublas Implementation machen:
#   - sparse matrizen von pyublas probiern
#   - mein makefile mal auf diese Liste posten
#   - die Beschreibung dann mal durchgehen
#   - den benchmark hier implementieren
#
# - dann unbedingt schaun wie man in python gscheit profiled
#
# - mit double und single precision profilen
#
# - pycuda mal probiern
#
# - cython nochmal versuchen ?

class ESNDummy():
    """ Simple ESN dummy for speed comparison.
    """
    
    def __init__(self,size=100,conn=1,inputs=1,outputs=1):
        self.N = size
        self.ins = inputs
        self.outs = outputs
        
        self.Wout = np.random.rand(self.outs, self.N+self.ins) * 2 - 1
        self.W = np.random.rand(self.N, self.N) * 2 - 1
        self.Win = np.random.rand(self.N, self.ins) * 2 - 1
        self.x = np.zeros((self.N))
        
        # for aureservoir
        self.aunet = au.DoubleESN()
        self.aunet.setSize(size)
        self.aunet.setInputs(inputs)
        self.aunet.setOutputs(outputs)
        self.aunet.setInitParam(au.ALPHA, 1.)
        self.aunet.setInitParam(au.CONNECTIVITY, conn)
        self.aunet.setInitParam(au.IN_CONNECTIVITY, 1.)
        self.aunet.init()
        self.aunet.setWout(self.Wout)
        self.aunet.post()
        
        # for sparse simulation
        self.aunet.getW(self.W)
        self.Wsp = sparse.csr_matrix( self.W ) # not efficient, use lil to construct matrix
        
        # create matrix for pysparse and pyublas.sparse
        tmp = spmatrix.ll_mat(self.N, self.N, int(self.N*self.N*conn) )
#        tmp2 = pyublas.zeros((self.N, self.N), flavor=pyublas.SparseBuildMatrix )
        for i in range(self.N):
            for j in range(self.N):
                if self.W[i,j] != 0:
                    tmp[i,j] = self.W[i,j]
#                    tmp2[i,j] = self.W[i,j]
        self.Wsp2 = tmp.to_csr()
#        self.Wsp3 = pyublas.asarray(tmp2, flavor=pyublas.SparseExecuteMatrix)
        
 
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
            self.x = self.Wsp * self.x
            self.x += np.dot(self.Win, indata[:, n])
            self.x = np.tanh(self.x)
            
            # calculate output
            outdata[:,n] = np.dot(self.Wout, np.r_[self.x,indata[:,n]] )
        
        return outdata

    def simulateAureservoir(self, indata):
        """ Standard Simulation without feedback, with aureservoir.
        """
        simsteps = indata.shape[1]
        outdata = np.zeros(indata.shape)
        
        self.aunet.simulate(indata,outdata)
        return outdata
    
    def simulatePysparse(self, indata):
        """ Standard Simulation without feedback, implemented with pysparse.
        """
        simsteps = indata.shape[1]
        outdata = np.zeros(indata.shape)
        t = self.x.copy()
        
        for n in range(simsteps):
            # calculate new internal state
            self.Wsp2.matvec(self.x,t)
            t += np.dot(self.Win, indata[:, n])
            self.x = np.tanh(t)
            
            # calculate output
            outdata[:,n] = np.dot(self.Wout, np.r_[self.x,indata[:,n]] )
        
        return outdata

    def simulatePyublasSparse(self, indata):
        """ Standard Simulation without feedback, implemented with pyublas sparse.
        """
        simsteps = indata.shape[1]
        outdata = np.zeros(indata.shape)
        t = self.x.copy()
        
        for n in range(simsteps):
            # calculate new internal state
            t = self.Wsp3 * self.x
            t += np.dot(self.Win, indata[:, n])
            self.x = np.tanh(t)
            
            # calculate output
            outdata[:,n] = np.dot(self.Wout, np.r_[self.x,indata[:,n]] )
        
        return outdata
    
    def simulateBlitz(self, indata):
        """ Standard Simulation, with weave.blitz.
        """
        simsteps = indata.shape[1]
        outdata = np.zeros(indata.shape)
        
        x = self.x
        W = self.W
        Win = self.Win
        Wout = self.Wout
        
        expr = """
for n in range(simsteps):
    x = np.dot(W, x)
    x += np.dot(Win, indata[:, n])
    x = np.tanh(x)
    outdata[:,n] = np.dot(Wout, np.r_[x,indata[:,n]] )
"""
        weave.blitz(expr, check_size=0)
        
        return outdata

    def simulateInline(self, indata):
        """ Standard Simulation without feedback, with weave.inline.
        ATTENTION: there is a problem with g++-4.3 !!
        """
        simsteps = indata.shape[1]
        outdata = np.zeros(indata.shape)
        
        x = self.x
        t = np.zeros(x.size)
        W = self.Wsp.data
        W_ind = self.Wsp.indices
        W_ptr = self.Wsp.indptr
        Win = self.Win
        Wout = self.Wout
        N = x.size
        
        code = """
        using namespace blitz;
        
        for(int n=0; n<simsteps; ++n)
        {
          for (int i=0 ; i <  N ; ++i)
          {
            t(i) =  0.0;

            // sparse matrix-vector product
            for (int j=W_ptr(i); j<W_ptr(i+1); ++j)
              t(i) += W(j) * x(W_ind(j));
            
            //self.x += np.dot(self.Win, indata[:, n])
            t(i) += Win(i,0) * indata(0,n);
            
            // self.x = np.tanh(self.x)
            t(i) = tanh( t(i) );
          }
          
          // outdata[:,n] = np.dot(self.Wout, np.r_[self.x,indata[:,n]] )
          // TODO
        }
        """
        
        err = weave.inline(code,
                           ['simsteps','x','t','W','W_ind','W_ptr','Win',
                            'indata','N'],
                           type_converters = converters.blitz,
                           compiler = 'gcc',
                           extra_compile_args=['-O3 -fPIC -ffast-math -mfpmath=sse -msse -msse2'])
        
        return outdata
