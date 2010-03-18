"""
Module for MDP Nodes that support gradient based learning.

This module contains the gradient node base class and some gradient based
implementations of MDP nodes.

"""

import mdp
import Engine
from mdp import numx
from mdp.utils import mult


class GradientExtensionNode(mdp.ExtensionNode, mdp.Node):
    """Base class for gradient based MDP nodes.

    The gradient method returns the gradient of a node based on its last input
    and output after a backpropagation sweep.  The params method returns the
    parameters of the node as a 1-d array.  The update_params method takes a
    1-d array as its argument and uses it to update the parameters of the node.
    """

    extension_name = "gradient"

    # Standard execute but it saves the data in self._last_x.
    def execute(self, x, *args, **kwargs):
        """Process the data contained in 'x'.
        
        If the object is still in the training phase, the function
        'stop_training' will be called.
        'x' is a matrix having different variables on different columns
        and observations on the rows.
        
        By default, subclasses should overwrite _execute to implement
        their execution phase. The docstring of the '_execute' method
        overwrites this docstring.
        """

        # TODO: I can't get passed the execution checks somehow because they
        # give errors about the node not being trainable.
        #self._pre_execution_checks(x)
        self._last_x = x
        self._last_y = self._execute(self._refcast(x), *args, **kwargs)
        return self._last_y

    def gradient(self):
        """Return the gradient that was found after the last backpropagation sweep."""
        return self._gradient()

    def params(self):
        """Return the parameters of the node as a 1-d array."""
        return self._params()

    def update_params(self, x):
        """Update the parameters of the node with a 1-d array."""
        self._update_params(x)

    def is_invertible(self):
        return True

    def is_trainable(self):
        return False

    def inverse(self, y):
        """Calls _gradient_inverse instead of the default _inverse."""
        return self._gradient_inverse(y)

    def param_size(self):
        """Return the total number of trainable parameters of the model."""
        return self._param_size()

    def is_training(self):
        return False

    ## Hook methods to be overwritten.

    def _gradient(self):
        pass

    def _params(self):
        pass

    def _update_params(self, x):
        pass

    def _gradient_inverse(self, x):
        pass

    def _param_size(self):
        pass



class BackpropNode(mdp.Node):
    """Node that handles backpropagation through a flow.
        
    It contains methods for obtaining gradients and loss values from a flow
    of gradient nodes.  It also has a trainer assigned that uses these
    methods for optimization of the parameters of all the nodes in the flow.
    """

    def __init__(self, gflow, gtrainer, loss_func=None, derror=None, dtype='float64'):

        self.gflow = gflow
        self.gtrainer = gtrainer
        self.loss_func = loss_func
        self.derror = derror

        if self.derror == None:
            self.derror = self.delta_error

        input_dim = gflow[0].get_input_dim()
        output_dim = gflow[-1].get_output_dim()

        super(BackpropNode, self).__init__(input_dim, output_dim, dtype)

    def delta_error(self, x, t):
        """Function that returns error gradient with respect to x."""
        return x - t

    def _train(self, x, **kwargs):
        """Update the parameters according to the input 'x' and target output 't'."""

        t = kwargs.get('t')
        # Enter gradient mode.
        mdp.activate_extension('gradient')

        # TODO: It is somehow a bit messy that the trainers already change the
        # parameters during their evaluation of the objective function but
        # their final solution might be different from the last parameter
        # vector they used to evaluate it...

        # Generate objective function for the current data.
        def func(params):
            return self._objective(x, t, params)

        update = self.gtrainer(func, self._params(), **kwargs)

        self._update_params(update)

        mdp.deactivate_extension('gradient')

    def _objective(self, x, t, params=None):
        """Get the gradient and loss of the objective.

        This method returns a tuple with the gradient as a 1-d array and the
        loss if available.  If params is defined it will first update the
        parameters.
        """

        # TODO: build checks to see if we are actually in gradient mode.
        # What if the top node is not a loss node?

        if not params == None:
            self._update_params(params)

        y = self.gflow.execute(x)
        
        if self.loss_func:
            loss = self.loss_func(y, t)
        else:
            loss = None

        delta = self.derror(y, t)

        self.gflow.inverse(delta)
        gradient = self._gradient()

        return (gradient, loss)

    def _gradient(self):
        """Get the gradient with respect to the parameters.

        This gradient has been calculated during the last backprop sweep.
        """

        gradient = numx.array([])

        for n in self.gflow:
            gradient = numx.concatenate((gradient, n.gradient()))

        return gradient

    def _params(self):
        """Return the current parameters of the nodes."""

        params = numx.array([])

        for n in self.gflow:
            params = numx.concatenate((params, n.params()))

        return params

    def _update_params(self, params):

        # Number of parameters we distributed so far.
        counter = 0

        for n in self.gflow:
            length = n._param_size()
            n.update_params(params[counter:counter + length])
            counter += length

    def _execute(self, x):
        return self.gflow.execute(x)

    def is_trainable(self):
        return True



## MDP (Engine) gradient node implementations ##

class GradientPerceptronNode(GradientExtensionNode,
                             Engine.nonlinear_nodes.PerceptronNode):
    """Gradient version of Engine Perceptron Node"""

    def _gradient(self):
        return self._gradient_vector

    def _params(self):
        return numx.concatenate((self.w.ravel(), self.b.ravel()))

    def _update_params(self, x):
        nw = self.w.size
        self.w.flat = x[:nw]
        self.b = x[nw:]

    def _gradient_inverse(self, y):
        x = self._last_x
        dy = self.transfer_derv(self._last_y) * y
        dw = mult(x.T, dy)
        self._gradient_vector = numx.concatenate((dw.ravel(), dy.sum(axis=0)))
        dx = mult(self.w, dy.T).T
        return dx

    def _param_size(self):
        return self.w.size + self.b.size

