'''
Created on Aug 20, 2009

@author: dvrstrae
'''
import numpy as np
import scipy as sp
import scipy.io as io
import mdp
import glob
import os


def narma30(n_samples = 10, sample_len = 1000):
    system_order = 30
    x, y = [], []
    for e in range(n_samples):
        x.append(np.random.rand(sample_len,1)*.5)
        x[e].shape = (-1,1)
        y.append(np.zeros((sample_len,1)))
        for k in range(system_order,sample_len-1):
            y[e][k+1]=.2*y[e][k]+.04*y[e][k]*np.sum(y[e][k-29:k])+1.5*x[e][k-29]*x[e][k] +.001
        
    return x,y 

def analog_speech (indir='../datasets/Lyon128', indices=range(501)):
    files= glob.glob(os.path.join(indir, '*.mat'))
    x,y=[],[]
    if len(files)>0:
        print "Found %d files in directory %s. Loading..." % (len(files), indir)
        for f in mdp.utils.progressinfo(files):
            contents = io.loadmat(f)
            x.append(contents['spec'].T)
            y.append(-1*np.ones([x[-1].shape[0],10]))
            # Fourth last character in filename indicates the digit class
            y[-1][:,f[-5]] = 1           
    else:
        print "No files found in %s" % (indir)
    return x,y
    pass