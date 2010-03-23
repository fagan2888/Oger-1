# Shows the use of the gradient based learning module.
import mdp
import pylab
import numpy as np
from mdp.utils import mult
from Engine.nonlinear_nodes import PerceptronNode
from Engine.gradient.gradient_nodes import BackpropNode, GradientPerceptronNode
from Engine.gradient.trainers import *
from Engine.utility_functions import *
from Engine.error_measures import mse

# Generate training data: a random sample from the sine function
data_in = np.random.uniform(-1, 1, 50)
data_in[0] = -1;
data_in[1] = 1;
data_in.shape = (50, 1)
data_out = np.sin(data_in * 5 * np.pi)

# Generate 1-3-1 MLP 
percnode1 = GradientPerceptronNode(1, 12, transfer_func=TanhFunction)
percnode2 = GradientPerceptronNode(12, 1)
myflow = percnode1 + percnode2

#bpnode = BackpropNode(myflow, gradient_descent)
#bpnode.train(x=data_in, t=data_out, epochs=1500, learning_rate=.001, momentum=.3)

choice = input('What optimization method do you want to use?\n0: Conjugate gradient\n1: BFGS (low memory Newton)\n2: gradient descent\n3: RPROP\n...')
choices = [CGTrainer(), BFGSTrainer(), GradientDescentTrainer(epochs=30000), RPROPTrainer(epochs=30000)]

bpnode = BackpropNode(myflow, choices[choice], loss_func=mse)
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

