# This file demonstrates the use of the CRBMNode (on a very trivial task...).

import pylab
import mdp
import mdp.hinet
import numpy as np
import Engine
from Engine.rbm_nodes import CRBMNode

# The IdentityNode is copied from a newer version of mdp
class IdentityNode(Node):
    """Return input data (useful in complex network layouts)"""
    def is_trainable(self):
        False

    def _set_input_dim(self, n): 
        self._input_dim = n 
        self._output_dim = n 


# Some recycled functions for data creation.


def generate_data(N):
    """Creates a noisy dataset with some simple pattern in it."""
    T = N * 38
    u = np.mat(np.zeros((T, 20)))
    for i in range(1, T, 38):
        if i % 76 == 1:
            u[i-1:i+19, :] = np.eye(20)
            u[i+18:i+38, :] = np.eye(20)[np.arange(19, -1, -1)]
            u[i-1:i+19, :] += np.eye(20)[np.arange(19, -1, -1)] 
        else:
            u[i-1:i+19, 1] = 1
            u[i+18:i+38, 8] = 1
    return u

def get_context(u, N=4):
    T, D = u.shape
    x = np.zeros((T, D * N))
    for i in range(N - 1, T):
        dat = u[i - 1, :]
        for j in range(2, N + 1):
            dat = np.concatenate((dat, u[i - j, :]), 1)
        x[i, :] = dat
    return x


u = np.array(generate_data(60))
t = np.zeros(u.shape)
t[:-1, :] = u[1:, :]

# Size of the context.
N = 12

epochs = 10

x = np.array(get_context(u, N))
x += np.random.normal(0, .001, x.shape)

# The context is concatenated to the input as if it where one signal.
v = np.concatenate((u, x), 1)

# Several nodes will be created to create a hierarchy of CRBMs that use context
# data coming from reservoirs. Moreover, one reservoir receives input that went
# throught a PCA node as well. This is solved by using layers and identity nodes.

# First reservoir layer
reservoir1 = Engine.reservoir_nodes.ReservoirNode(input_dim=20, output_dim=300)
identity1 = mdp.nodes.IndentiyNode(input_dim=20)

ReservoirLayer1 = mdp.hinet.SameInputLayer(reservoir1 + identity1)

# First CRBM.
# Note that the output of the InputLayer will be 320 dimensional.
crbmnode1 = CRBMNode(hidden_dim=300, visible_dim=20, context_dim=300)

# PCA layer
pcanode = mdp.nodes.PCANode(input_dim=300, output_dim=20)
identity2 = mdp.nodes.IndentiyNode(input_dim=300)

PCALayer = mdp.hinet.SameInputLayer(pcanode + identity2)

# Second reservoir layer.
# Note that this time a normal layer is used to split the PCA and raw data.
reservoir2 = Engine.reservoir_nodes.ReservoirNode(input_dim=20, output_dim=300)
identity3 = mdp.nodes.IndentiyNode(input_dim=300)

ReservoirLayer2 = mdp.hinet.Layer(reservoir1 + identity1)

# Second CRBM.
crbmnode2 = CRBMNode(hidden_dim=300, visible_dim=300, context_dim=300)

# And finally a linear classifier to put on top.
readout = Engine.linear_nodes.RidgeRegressionNode(input_dim=300, output_dim=20)

theflow = ReservoirLayer1 + crbmnode1 + PCALayer + ReservoirLayer2 + crbmnode2 + readout

theflow.train(v, t)

#
#crbmnode.stop_training()
#
#hiddens, sampl_h = crbmnode.sample_h(u, x)
#
#print 'Sampling...'
#v_zero = np.random.normal(0, 1, u.shape)
#for i in range(25):
#    visibles, sampl_v = crbmnode.sample_v(sampl_h, x)
#    hiddens, sampl_h = crbmnode.sample_h(sampl_v, x)
#
#visibles, sampl_v = crbmnode.sample_v(sampl_h, x)
#error = np.mean((u.ravel() - visibles.ravel())**2)
#print 'Final MSE:', error
#
#pylab.clf()
#p = pylab.subplot(411)
#p = pylab.imshow(u[:500, :].T)
#p = pylab.subplot(412)
#p = pylab.imshow(visibles[:500, :].T)
#p = pylab.show()
#
