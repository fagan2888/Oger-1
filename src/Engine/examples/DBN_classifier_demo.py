import mdp
import numpy as np
import cPickle
from Engine.gradient.trainers import *
from Engine.utility_functions import *
from Engine.gradient.gradient_nodes import *
from Engine.error_measures import ce
from Engine.nonlinear_nodes import PerceptronNode
from Engine.rbm_nodes import RBMNode

data = cPickle.load(open('/home/pbrakel/Data/mnist/mnist.p'))

n_train = 200
n_test = 100
epochs = 25

image_data = data['trainimages']
image_labels = data['trainlabels']

image_data /= 255.

train_data = image_data[:n_train]
train_labels = image_labels[:n_train]

test_data = image_data[n_train:n_train + n_test]
test_labels = image_labels[n_train:n_train + n_test]

# Generate a small subset of the data.

rbmnode1 = RBMNode(784, 100)
rbmnode2 = RBMNode(100, 200)
percnode = PerceptronNode(200, 10, transfer_func=SoftmaxFunction)

# Greedy pretraining of RBMs

for epoch in range(epochs):
    for c in train_data:
        rbmnode1.train(c.reshape((1, 784)), n_updates=1, epsilon=.1)

hiddens = rbmnode1(train_data)

for epoch in range(epochs):
    for c in hiddens:
        rbmnode2.train(c.reshape((1, 100)), n_updates=1, epsilon=.1)

# Create flow and backpropagation node.


myflow = rbmnode1 + rbmnode2 + percnode

bpnode = BackpropNode(myflow, GradientDescentTrainer(momentum=.9), loss_func=ce)

# Fine-tune for classification
for epoch in range(epochs):
    for i in range(len(train_data)):
        label = np.array(np.eye(10)[train_labels[i], :])
        bpnode.train(x=train_data[i].reshape((1, 784)), t=label.reshape((1, 10)))

# Evaluate performance on test set.
out = bpnode(test_data)

out[np.arange(out.shape[0]), np.argmax(out, axis=1)] = 1
out[out < 1] = 0
t_test = np.array([int(i) for i in test_labels])
correct = np.sum(out[np.arange(len(t_test)), t_test])
print correct / float(len(test_labels))



