"""
Module for MDP Nodes that support gradient based learning.

This module contains the gradient node base class and some gradient based
implementations of MDP nodes.

"""

import mdp
import Engine
from mdp import numx
from mdp.utils import mult

# TODO: would we ever plan to use Hessians?
class GradientNode(mdp.Node):
    def _gradient(self):
        return self._gradient_vector

    def _params(self):
        pass

    def _set_params(self, x):
        pass

    # TODO: rename to calculate_gradient?
    def _gradient_inverse(self, x):
        pass

    def _param_size(self):
        pass


    #TODO: why do we need this?
    def gradient(self):
        """Return the gradient that was found after the last backpropagation sweep."""
        return self._gradient()

    #TODO: why do we need this?
    def params(self):
        """Return the parameters of the node as a 1-d array."""
        return self._params()

    #TODO: why do we need this?
    def set_params(self, x):
        """Update the parameters of the node with a 1-d array."""
        self._set_params(x)

# TODO: why does it break if we subclass GradientNode instead of Node?
class GradientExtensionNode(mdp.ExtensionNode, mdp.Node):
    """Base class for gradient based MDP nodes.

    The gradient method returns the gradient of a node based on its last input
    and output after a backpropagation sweep.  The params method returns the
    parameters of the node as a 1-d array.  The set_params method takes a
    1-d array as its argument and uses it to update the parameters of the node.
    """

    extension_name = "gradient"

    # Standard execute but it saves the data in self._last_x.
    # TODO: why can't this be _execute ??
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

        self._last_x = x
        self._last_y = self._non_extension_execute(x, *args, **kwargs)
        return self._last_y

    def inverse(self, y):
        """Calls _gradient_inverse instead of the default _inverse."""
        return self._gradient_inverse(y)

    def is_invertible(self):
        return True

    def is_trainable(self):
        return False

    def is_training(self):
        return False


class BackpropNode(mdp.Node):
    """Node that handles backpropagation through a flow.
        
    It contains methods for obtaining gradients and loss values from a flow
    of gradient nodes.  It also has a trainer assigned that uses these
    methods for optimization of the parameters of all the nodes in the flow.
    """

    def __init__(self, gflow, gtrainer, loss_func=None, derror=None, dtype='float64'):

        self.gflow = gflow
        self.gtrainer = gtrainer
        
        # TODO: can this combination be in an object
        self.loss_func = loss_func
        self.derror = derror

        if self.derror == None:
            self.derror = lambda x, t: x - t

        input_dim = gflow[0].get_input_dim()
        output_dim = gflow[-1].get_output_dim()

        super(BackpropNode, self).__init__(input_dim, output_dim, dtype)

    def _train(self, x, **kwargs):
        """Update the parameters according to the input 'x' and target output 't'."""

        # Extract target values and delete them so they don't interfere with
        # the train method call below.
        # TODO: Perhaps the use of target values should be optional to allow
        # for unsupervised algorithms etc.
        t = kwargs.get('t')
        del kwargs['t']
        # Enter gradient mode.
        mdp.activate_extension('gradient')

        # Generate objective function for the current data.
        def func(params):
            return self._objective(x, t, params)

        update = self.gtrainer.train(func, self._params(), **kwargs)

        self._set_params(update)

        mdp.deactivate_extension('gradient')

    def _objective(self, x, t, params=None):
        """Get the gradient and loss of the objective.

        This method returns a tuple with the gradient as a 1-d array and the
        loss if available.  If params is defined it will first update the
        parameters.
        """

        if not params == None:
            self._set_params(params)

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

    def _set_params(self, params):

        # Number of parameters we distributed so far.
        counter = 0

        for n in self.gflow:
            length = n._param_size()
            n.set_params(params[counter:counter + length])
            counter += length

    def _execute(self, x):
        return self.gflow.execute(x)

    def is_trainable(self):
        return True



## MDP (Engine) gradient node implementations ##

# Should this not just be part of the PerceptronNode?
class GradientPerceptronNode(GradientNode, Engine.nodes.PerceptronNode):
    """Gradient version of Engine Perceptron Node"""

    def _params(self):
        return numx.concatenate((self.w.ravel(), self.b.ravel()))

    def _set_params(self, x):
        nw = self.w.size
        self.w.flat = x[:nw]
        self.b = x[nw:]

    def _gradient_inverse(self, y):
        x = self._last_x
        dy = self.transfer_func.df(x, self._last_y) * y
        dw = mult(x.T, dy)
        self._gradient_vector = numx.concatenate((dw.ravel(), dy.sum(axis=0)))
        dx = mult(self.w, dy.T).T
        return dx

    def _param_size(self):
        return self.w.size + self.b.size
    
class GradientReservoirNode(GradientNode, Engine.nodes.ReservoirNode):

    # TODO: should this parameter be called y?
    def _gradient_inverse(self, y):
        x = self._last_x
        dy = self.transfer_func.df(x, self._last_y) * y
        dw = mult(x.T, dy)
        self._gradient_vector = numx.concatenate((dw.ravel(), dy.sum(axis=0)))
        dx = mult(self.w, dy.T).T
        return dx

    def _param_size(self):
        return 0

class GradientRBMNode(GradientNode, Engine.nodes.ERBMNode):
    """Gradient version of the Engine RBM Node.

    This gradient node is intended for use in a feedforward architecture. This
    means that the biases for the visibles are ignored.
    """

    def _params(self):
        return numx.concatenate((self.w.ravel(), self.bh.ravel()))

    def _set_params(self, x):
        nw = self.w.size
        self.w.flat = x[:nw]
        self.b = x[nw:]

    def _gradient_inverse(self, y):
        x = self._last_x
        dy = Engine.utils.LogisticFunction.df(x, self._last_y) * y
        dw = mult(x.T, dy)
        self._gradient_vector = numx.concatenate((dw.ravel(), dy.sum(axis=0)))
        dx = mult(self.w, dy.T).T
        return dx

    def _param_size(self):
        return self.w.size + self.bh.size

    def inverse(self, y):
        """Calls _gradient_inverse instead of the default _inverse."""
        return self._gradient_inverse(y)

