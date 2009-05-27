from mdp import numx, numx_linalg, utils, Node, NodeException, TrainingException
from mdp.utils import mult
import mdp


class RidgeRegressionNode(Node):
    """Least-square ridge regression node.
    """

    def __init__(self, ridge_param, input_dim=None, output_dim=None, dtype=None):
        """
        """
        super(RidgeRegressionNode, self).__init__(input_dim, output_dim, dtype)
        
        # for the linear regression estimator we need two terms
        # the first one is X^T X
        self._xTx = None
        # the second one is X^T Y
        self._xTy = None

        # keep track of how many data points have been sent
        self._tlen = 0

        # final regression coefficients
        self.weight = None
        self.ridge_param = ridge_param

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def is_invertible(self):
        return False

    def _check_train_args(self, x, y):
        # set output_dim if necessary
        if self._output_dim is None:
            self._set_output_dim(y.shape[1])
        # check output dimensionality
        self._check_output(y)
        if y.shape[0] != x.shape[0]:
            msg = ("The number of output points should be equal to the "
                   "number of datapoints (%d != %d)" % (y.shape[0], x.shape[0]))
            raise TrainingException(msg)

    def _train(self, x, y):
        # initialize internal vars if necessary
        if self._xTx is None:
            x_size = self._input_dim+1
            self._xTx = numx.zeros((x_size, x_size), self._dtype)
            self._xTy = numx.zeros((x_size, self._output_dim), self._dtype)

        x = self._add_constant(x)

        # update internal variables
        self._xTx += mult(x.T, x)
        self._xTy += mult(x.T, y)
        self._tlen += x.shape[0]

    def _stop_training(self):
        try:
            inv_xTx = utils.inv(self._xTx + self.ridge_param * mdp.numx.eye(self._input_dim+1))
        except numx_linalg.LinAlgError, exception:
            errstr = (str(exception) + 
                      "\n Input data may be redundant (i.e., some of the " +
                      "variables may be linearly dependent).")
            raise NodeException(errstr)

        self.weight = mult(inv_xTx, self._xTy)

    def _execute(self, x):
        x = self._add_constant(x)
        return mult(x, self.weight)

    def _add_constant(self, x):
        """Add a constant term to the vector 'x'.
        x -> [1 x]
        """
        return numx.concatenate((numx.ones((x.shape[0], 1),
                                           dtype=self.dtype), x), axis=1)

class WashoutNode(Node):
    """ remove initial states.
    """

    def __init__(self, washout, input_dim=None, dtype='float64'):
        super(WashoutNode, self).__init__(input_dim, input_dim, dtype)
        self.washout = washout

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x):
        return x[self.washout:,:]

# This is a small test
if __name__ == '__main__':
  import mdp	
  print "let's go!!"
  node = RidgeRegressionNode(1e5)
  node.train(mdp.numx_rand.randn(1000,10), mdp.numx_rand.randn(1000,1))
  node.train(mdp.numx_rand.randn(1000,10), mdp.numx_rand.randn(1000,1))
  node.stop_training()
  print node.weight
  node.execute(mdp.numx_rand.randn(1000,10))
  print "using a flow"
  flow = mdp.Flow([WashoutNode(10), RidgeRegressionNode(10)])
  flow.train([None, [(mdp.numx_rand.randn(1000,10), mdp.numx_rand.randn(990,1))]])
  flow.train([None, [(mdp.numx_rand.randn(1000,10), mdp.numx_rand.randn(990,1))]])
  flow.execute(mdp.numx_rand.randn(1000,10))
  print flow[1].weight
