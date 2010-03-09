'''
Created on Aug 24, 2009

@author: dvrstrae
'''
import numpy as np
import mdp
from mdp.utils import mult


class SignNode(mdp.Node):
    """
     Compute the sign function of the input.
        
        This simple node computes the sign function of its input
    
    # PyUML: Do not remove this line! # XMI_ID:__6oqMK0EEd6_mKvLwRcUQA
    """
    
    def __init__(self, input_dim=None, output_dim=None, dtype='float64'):
        
        super(SignNode, self).__init__(input_dim, output_dim, dtype)
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        return np.sign(x)


class PerceptronNode(mdp.Node):
    """
    Trains a (non-linear) perceptron using gradient descent.

    The output transfer function can be specified together with the appropriate
    derivative for the input (only needed for back propagation). 

    Using softmax as the transfer function turns this into logistic regression.
    """

    def __init__(self, input_dim=None, output_dim=None, transfer_func=None,
                 transfer_derv=None, dtype='float64'):
        
        super(PerceptronNode, self).__init__(input_dim, output_dim, dtype)
        self.w = self._refcast(mdp.numx.random.randn(self.input_dim, self.output_dim)*0.01)
        self.b = self._refcast(mdp.numx.random.randn(self.output_dim)*0.01)

        if transfer_func == None:
            self.transfer_func = lambda x: x
        else:
            self.transfer_func = transfer_func

        if transfer_derv == None:
            self.transfer_derv = lambda x: 1.0
        else:
            self.transfer_derv = transfer_derv

        self._delta = (0., 0.)

    def train(self, x, t, n_epochs=1, epsilon=0.1, decay=0., momentum=0.,
              batch_size=1, verbose=False):
        """Update the parameters according to the input 'v' and target output 't'.

        The training is performed using gradient descent.

        x -- a matrix having different variables on different columns and
             observations on the rows.
        n_updates -- number of training epochs.
        epsilon -- learning rate. Default value: 0.1
        decay -- weight decay term. Default value: 0.
        momentum -- momentum term. Default value: 0.
        batch_size -- datapoints to process simultaneously. Default value: 1
        """

        if not self.is_training():
            errstr = "The training phase has already finished."
            raise mdp.TrainingFinishedException(errstr)

        self._check_input(x)

        self._train_phase_started = True
        self._train(x, t, n_epochs, epsilon, decay, momentum, batch_size, verbose)

    def _train(self, x, t, n_epochs=1, epsilon=0.1, decay=0., momentum=0.,
              batch_size=1, verbose=False):
        """Update the parameters according to the input 'v' and target output 't'.

        The training is performed using gradient descent.

        x -- a matrix having different variables on different columns and
             observations on the rows.
        n_updates -- number of training epochs.
        epsilon -- learning rate. Default value: 0.1
        decay -- weight decay term. Default value: 0.
        momentum -- momentum term. Default value: 0.
        batch_size -- datapoints to process simultaneously. Default value: 1
        """

        # Useful quantities.

        uW, ub = self._delta
        n, d = x.shape

        assert(n == t.shape[0])

        for epoch in range(n_epochs):
            for i in range(0, n, batch_size):
                start = i
                if n - i * batch_size < batch_size:
                    stop = n - i
                else:
                    stop = i + batch_size

                dW, db = self.get_gradient(x[start:stop], t[start:stop])
                
                uW = momentum * uW + epsilon * dW - decay * self.w
                ub = momentum * ub + epsilon * db
                self.w += uW
                self.b += ub

    def get_gradient(self, x, t):
        """Get the error negative gradient with respect to the weights and biases.
        """

        out = self.execute(x)
        dout = out - t
        dW = -mult(x.T, dout)
        db = -dout.sum(axis=0)

        return dW, db

    def _up_pass(self, x, epsilon=.01, decay=0.0, momentum=0.0):
        """Same as execute but doesn't require training to be complete."""
        y = self.transfer_func(mult(x, self.w) + self.b)
        self._orig_x = x  # Store input
        return y

    def _down_pass(self, e, epsilon=0.1, decay=0.0,
                  momentum=0.0):
        """Pass error back to previous layer after applying derivative."""

        x = self._orig_x
        uW, ub = self._delta
        n, d = x.shape

        dW = -mult(self._orig_x.T, d)
        db = x
        e = mult(self.w.T, e)
                
        uW = momentum * uW + epsilon * dW - decay * self.w
        ub = momentum * ub + epsilon * db
        self.w += uW
        self.b += ub
        return e

    def is_trainable(self):
        return True
    
    def is_invertible(self):
        return False
    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):

        y = self.transfer_func(mult(x, self.w) + self.b)
        return y


