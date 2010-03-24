# Shows the use of the gradient based learning module.
import pylab
from Engine.nonlinear_nodes import PerceptronNode
from Engine.gradient.gradient_nodes import BackpropNode 
from Engine.gradient.trainers import *
from Engine.utility_functions import *
from Engine.error_measures import mse

# Generate training data: a random sample from the sine function
data_in = np.random.uniform(-1, 1, 50)
#data_in = np.arange(-1, 1, .11)
data_in.shape += (1,)
data_out = np.sin(data_in * 5 * np.pi)

# Generate 1-12-1 MLP 
percnode1 = PerceptronNode(1, 12, transfer_func=TanhFunction)
percnode2 = PerceptronNode(12, 1)
myflow = percnode1 + percnode2

choice = input("""What optimization method do you want to use?
0: Conjugate gradient
1: BFGS (low memory Newton)
2: gradient descent
3: RPROP
4: L-BFGS-B
...""")
choices = [CGTrainer(), BFGSTrainer(), GradientDescentTrainer(epochs=30000), RPROPTrainer(epochs=30000),LBFGSBTrainer(weight_bounds=(-10,10))]

bpnode = BackpropNode(myflow, choices[choice], loss_func=mse)
bpnode.train(x=data_in, t=data_out)

data_in2 = np.arange(-2, 2, .005)
data_in2.shape += (1,)
data_out2 = np.sin(data_in2 * 5 * np.pi)

out = bpnode(data_in2)

print "Done!"

pylab.scatter(data_in, np.array(data_out))
pylab.plot(data_in2, out, data_in2, data_out2)
pylab.show()

