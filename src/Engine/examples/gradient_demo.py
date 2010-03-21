# Shows the use of the gradient based learning module.
import mdp
import pylab
import numpy as np
from mdp.utils import mult
from Engine.nonlinear_nodes import PerceptronNode
from Engine.gradient.gradient_nodes import BackpropNode
from Engine.gradient.trainers import gradient_descent
from scipy.optimize import *

def cg_trainer(func, x0):
    fobj = lambda x: func(x)[1]
    fprime = lambda x: func(x)[0]
    return fmin_cg(fobj, x0, fprime)

def bfgs_trainer(func, x0):
    fobj = lambda x: func(x)[1]
    fprime = lambda x: func(x)[0]
    return fmin_bfgs(fobj, x0, fprime, disp=2)

def mse_loss(x, t):
    return sum((x - t)**2)

data_in = np.random.uniform(-1, 1, 50)
data_in.shape = (50, 1)
data_out = np.sin(data_in * 5 * np.pi)

percnode1 = PerceptronNode(1, 8, transfer_func=np.tanh,
                          transfer_derv=lambda x: 1-x**2)
percnode2 = PerceptronNode(8, 1)

myflow = percnode1 + percnode2

#bpnode = BackpropNode(myflow, gradient_descent)
#bpnode.train(x=data_in, t=data_out, epochs=1500, learning_rate=.001, momentum=.3)

choice = input('What optimization method do you want to use?\n0: Conjugate gradient\n1: BFGS (low memory Newton)\n...')
choices = [cg_trainer, bfgs_trainer]

bpnode = BackpropNode(myflow, choices[choice], loss_func=mse_loss)
bpnode.train(x=data_in, t=data_out)

data_in2 = np.random.uniform(-1, 1, 500)
data_in2.shape = (500, 1)
data_out2 = np.sin(data_in2 * 5 * np.pi)

out = bpnode(data_in2)

pylab.subplot(311)
pylab.scatter(data_in, np.array(data_out))
pylab.title('Train data')
pylab.subplot(312)
pylab.scatter(data_in2, out)
pylab.title('Network output')
pylab.subplot(313)
pylab.scatter(data_in2, data_out2)
pylab.title('True function')
pylab.show()

