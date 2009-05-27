#===============================================================================
# benchmarks to compair sparse matrix libraries
#===============================================================================

import numpy as np
from scipy import sparse, weave
from scipy.weave import converters
import random
import time

def simulate(func,args):
    """ benchmarks the given function
    """
    t = time.clock()
    res = func(*args)
    ttime = time.clock() - t
    return (ttime, res)

def random_matrix(size,conn):
    """ Generates a dense prototype matrix.
    """
    mtx = np.zeros((size,size))
    
    for i in range(size):
        for j in range(size):
            tmp = random.random()
            if tmp < conn:
                mtx[i,j] = 1 - tmp
    return mtx

def mv_dense(mtx,vec,rep):
    """ Dense Matrix-Vector Product for comparison.
    """
    for n in range(rep):
        res = np.dot(mtx, vec)
    return res

def mv_scipy(mtx,vec,rep):
    """ Matrix-Vector Product in Scipy.
    """
    for n in range(rep):
        res = mtx * vec
#        res = mtx.matvec(vec)
    return res

def mv_sparse_weave(mtx,vec,rep):
    """ Sparse matrix vector product for compressed row storage.
    """
    # compressed row storage: http://www.cs.utk.edu/~dongarra/etemplates/node373.html
    # matrix vec product: http://www.cs.utk.edu/~dongarra/etemplates/node382.html
    
    n = mtx.shape[0] # output dimension size
    y = np.zeros(n)
    a = mtx.data
    ind = mtx.indices
    ptr = mtx.indptr
    
    # sparse mv in python
#    for i in range(n):
#        #y[i] = 0.  # if y is not initialized to 0 !
#        for j in range(ptr[i], ptr[i+1]):
#            y[i] += a[j] * vec[ind[j]]
    
    # weave inline implementation
    code = """
        while(rep--)
        {
            for(int i=0; i<n; ++i) {
            y(i) = 0;
            for(int j=ptr(i); j<ptr(i+1); ++j) {
                y(i) += a(j) * vec(ind(j));
            } }
        }
        """
    err = weave.inline(code,
                       ['a','ind','ptr','vec','y','n','rep'],
                       type_converters = converters.blitz,
                       compiler = 'gcc',
                       extra_compile_args=['-O3 -fPIC -ffast-math -mfpmath=sse -msse -msse2'])
    return y


#===============================================================================
# MAIN function
#===============================================================================

if __name__ == "__main__":
    mtx_size = 1000
    mtx_conn = 0.1
    rep = 1000   # repetitions of the simulation
    
    # for repeating benchmarks
    random.seed(3)
    
    # dense prototype matrix and vector
    Mtx = random_matrix(mtx_size,mtx_conn)
    vec = np.ones(mtx_size)
    
    # init sparse matrices
    Mtx_scipy = sparse.csr_matrix(Mtx)
    
    print "\nCPU Time for Matrix-Vector Product:"
    print "-------------------------------------"
    print "numpy dense:\t", simulate(mv_dense, (Mtx,vec,rep))[0]
    print "scipy sparse:\t", simulate(mv_scipy, (Mtx_scipy,vec,rep))[0]
    print "weave inline:\t", simulate(mv_sparse_weave, (Mtx_scipy,vec,rep))[0]
    
    