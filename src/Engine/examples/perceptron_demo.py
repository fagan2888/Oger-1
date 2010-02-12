# Shows the limitation of a linear perceptron. I used two outputs for debug purposes.
import mdp
import pylab
import numpy as np
from mdp.utils import mult
from Engine.nonlinear_nodes import PerceptronNode

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

# Learn OR
data_in = np.zeros((4, 2))
data_in[1, :] = [0.0, 0.0]
data_in[1, :] = [1.0, 0.0]
data_in[2, :] = [0.0, 1.0]
data_in[3, :] = [1.0, 1.0]

data_out = np.zeros((4, 2))
data_out[0, :] = [0.0, 1.0]
data_out[1, :] = [1.0, 0.0]
data_out[2, :] = [1.0, 0.0]
data_out[3, :] = [1.0, 0.0]

percnode = PerceptronNode(2, 2, transfer_func=softmax)

percnode.train(data_in, data_out, n_epochs=30000, epsilon=1, momentum=.0)
out = percnode(data_in)
out = ((out.T == np.max(out, 1)) * 1).T

print 'OR'
print data_in
print out
print 'Correct output'
print data_out

# Learn XOR
data_in = np.zeros((4, 2))
data_in[1, :] = [1.0, 0.0]
data_in[2, :] = [0.0, 1.0]
data_in[3, :] = [1.0, 1.0]

data_out = np.zeros((4, 2))
data_out[0, :] = [0.0, 1.0]
data_out[1, :] = [1.0, 0.0]
data_out[2, :] = [1.0, 0.0]
data_out[3, :] = [0.0, 1.0]

percnode = PerceptronNode(2, 2, transfer_func=softmax)

percnode.train(data_in, data_out, n_epochs=30000, epsilon=1, momentum=.0)
out = percnode(data_in)
out = ((out.T == np.max(out, 1)) * 1).T

print 'XOR'
print data_in
print out
print 'Correct output'
print data_out

