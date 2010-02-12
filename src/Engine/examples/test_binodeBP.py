# This demo shows how the PerceptonBiNode can be used for batch gradient
# descent training of a multilayer perceptron. Note that stochastic gradient
# descent is also possible. The use of more efficient methods like BFGS is not
# implemented yet.

import numpy as np
import binet
import pylab
from Engine.binodes import PerceptronBiNode

if __name__ == '__main__':

    # Some training parameters.
    N_train = 100
    N_test = 500
    n_epochs = 15000
    hidden = 15
    lr = .001

    # Datasets to choose from.
    funcs = [lambda x: np.sin(x * 2 * np.pi)]
    funcs += [np.sign]
    funcs += [abs]
    funcs += [lambda x: x**2]

    choice = input("""Choose a function to train on:
                   0: sin(2 * pi * x)
                   1: sign(x)
                   2: abs(x)
                   3: x^2\n\r->""")

    func = funcs[choice]


    data_in = np.random.uniform(-1, 1, N_train)
    data_out = func(data_in)
    data_in.shape = (N_train, 1)
    data_out.shape = (N_train, 1)

    in_old = data_in.copy()
    out_old = data_out.copy()

    # First tanh layer.
    Layer1 = PerceptronBiNode('bottom', 1, hidden, transfer_func=np.tanh,
                              transfer_derv=lambda x: 1 - x**2)

    Layer1.W *= 100

    # Top layer that contains the desired output as 'error_func'.
    Layer2 = PerceptronBiNode('top', hidden, 1, transfer_func=np.tanh,
                              error_func=data_out)
    Layer2.W *= 10

    flow = binet.BiFlow([Layer1, Layer2])

    print 'Training...'
    flow.train(data_in, {"momentum" : 0.9, "decay" : lr * .02, "n_epochs" :
                         n_epochs, "epsilon" : lr})

    data_in = np.random.uniform(-1, 1, N_test)
    data_in.shape = (N_test, 1)

    out = flow(data_in)

    p = pylab.subplot(311)
    p = pylab.scatter(data_in, func(data_in))
    p = pylab.subplot(312)
    p = pylab.scatter(in_old, out_old)
    p = pylab.subplot(313)
    p = pylab.scatter(data_in, out)
    p = pylab.show()

