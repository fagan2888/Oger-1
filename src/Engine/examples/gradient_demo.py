# Shows the use of the gradient based learning module.
import mdp
import pylab
import numpy as np
from mdp.utils import mult
from Engine.nonlinear_nodes import PerceptronNode
from Engine.gradient.gradient_nodes import BackpropNode
from Engine.gradient.trainers import gradient_descent

def softmax(x):
    n, d = x.shape
    y = np.zeros(x.shape)
    for i in range(n): 
        y[i, :] = x[i, :] - max(x[i, :].ravel())  # Overflow protection.
        y[i, :] = np.exp(y[i, :]) / sum(np.exp(y[i, :]).ravel())
    return y

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

percnode1 = PerceptronNode(2, 8, transfer_func=np.tanh,
                          transfer_derv=lambda x: 1-x**2)
percnode2 = PerceptronNode(8, 2, transfer_func=softmax)

myflow = percnode1 + percnode2

bpnode = BackpropNode(myflow, gradient_descent)



bpnode.train(x=data_in, t=data_out, epochs=5000, learning_rate=.1)
out = bpnode(data_in)
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


percnode1 = PerceptronNode(2, 10, transfer_func=np.tanh,
                          transfer_derv=lambda x: 1-x**2)
percnode1.w *= 10
percnode2 = PerceptronNode(10, 2, transfer_func=softmax)

myflow = percnode1 + percnode2

bpnode = BackpropNode(myflow, gradient_descent)

bpnode.train(x=data_in, t=data_out, epochs=5000, learning_rate=.1, momentum=.1)

out = bpnode(data_in)
out = ((out.T == np.max(out, 1)) * 1).T

print 'XOR'
print data_in
print out
print 'Correct output'
print data_out

