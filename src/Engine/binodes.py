import mdp
from mdp.utils import mult
import binet
from binet.biflow import EXIT_TARGET
from Engine.nonlinear_nodes import PerceptronNode

class PerceptronBiNode(binet.BiNode, PerceptronNode):
    """Turns the PerceptronNode into a Binode for backpropagation learning."""

    def __init__(self, node_id, input_dim, output_dim, transfer_func=None,
                 transfer_derv=None, error_func=None, dtype='float64'):

        super(PerceptronBiNode, self).__init__(node_id=node_id,
                                               input_dim=input_dim,
                                               output_dim=output_dim,
                                               transfer_func=transfer_func,
                                               transfer_derv=transfer_derv,
                                               dtype=dtype)
        self._orig_x=None
        self._trainer = False
        self._iteration = 0

        # The error_func argument determines the role of the node. If it refers
        # to desired data, it is assumed that this is the top-node in the
        # hierarchy.
        self.error_func = error_func

    def _train(self, x, msg=None, epsilon=.1,
               momentum=0., decay=0., n_epochs=1):
        """Train the node using batch gradient descent on the data in 'x'.

        The desired data should be given to the top node in the hierarchy.
        """

        if self._iteration >= n_epochs:
            return

        # This node will manage the training cycles.
        self._trainer = True
        if msg == None:
            # Store parameters.
            msg = {}
            msg["epsilon"] = epsilon
            msg["momentum"] = momentum
            msg["decay"] = decay
            msg["n_epochs"] = n_epochs


        # Do the first up_pass
        msg[self.node_id + "=>method"] = "up_pass"
        target = 0


        return x, msg, target 
                    
    def _up_pass(self, x, msg=None, n_epochs=1):
        """Same as execute but doesn't require training to be complete.
        
        This method also manages backpropagation and should only be called by
        _train.
        """

        if self._iteration >= n_epochs and self._trainer:
            return x, None, 0

        if x == None:  # Bottom node...
            x = self._orig_x
        else:
            self._orig_x = x  # Store input

        y = self.transfer_func(mult(x, self.w) + self.b)
        self._last_y = y

        if self.error_func != None:  # This is the top node.
            error = y - self.error_func
            
            # Go down
            msg["method"] = "down_pass"
            y = error
            target = 0
        else: # Do up_pass on for the next node.
            msg["method"] = "up_pass"
            target = 1
        self._iteration += 1
        return y, msg, target

    def _down_pass(self, e, msg=None, epsilon=0.1, decay=0.0,
                  momentum=0.0):
        """Pass error back to previous layer after applying derivative."""

        x = self._orig_x
        uW, ub = self._delta
        n, d = x.shape

        e = e * self.transfer_derv(self._last_y)

        dW = mult(self._orig_x.T, e)
        db = e
                
        uW = momentum * uW - epsilon * dW - decay * self.w

        ub = momentum * ub - epsilon * db.sum(axis=0) / n
        self.w += uW
        self.b += ub


        if self._trainer:  # This must be the bottom node.
            # Do up pass on the current node using the original data.
            msg["method"] = "up_pass"
            y = self._orig_x
            target = 0
        else:
            # Go down further
            msg["method"] = "down_pass"
            target = -1
            y = mult(db, self.w.T)

        return y, msg, target


