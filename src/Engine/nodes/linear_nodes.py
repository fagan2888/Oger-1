'''
Created on Aug 20, 2009

@author: dvrstrae
'''

import mdp

class RidgeRegressionNode(mdp.Node):
    """
    Least-square ridge regression node.
    
    # PyUML: Do not remove this line! # XMI_ID:__6UhIK0EEd6_mKvLwRcUQA
    """

    def __init__(self, ridge_param=0, washout = 0, use_bias = 1, input_dim=None, output_dim=None, dtype=None):
        """
        """
        super(RidgeRegressionNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        
        # for the linear regression estimator we need two terms
        # the first one is X^T X
        self._xTx = None
        # the second one is X^T Y
        self._xTy = None

        # keep track of how many data points have been sent
        self.n_timesteps_accumulated = 0

        # final regression coefficients
        self.weights = None
        self.ridge_param = ridge_param
        self.washout = washout

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
            raise mdp.TrainingException(msg)

    def _train(self, x, y):
        # initialize internal vars if necessary
        if self._xTx is None:
            x_size = self._input_dim+1
            self._xTx = mdp.numx.zeros((x_size, x_size), self._dtype)
            self._xTy = mdp.numx.zeros((x_size, self._output_dim), self._dtype)

        x = self._add_constant(x)
        
        # update internal variables
        self._xTx += mdp.utils.mult(x.T, x)
        self._xTy += mdp.utils.mult(x.T, y)
        self.n_timesteps_accumulated += x.shape[0]

    def _stop_training(self):
        try:
            inv_xTx = mdp.utils.inv(self._xTx + self.ridge_param * mdp.numx.eye(self._input_dim+1))
        except mdp.numx_linalg.LinAlgError, exception:
            errstr = (str(exception) + 
                      "\n Input data may be redundant (i.e., some of the " +
                      "variables may be linearly dependent).")
            raise mdp.NodeException(errstr)
        self.weights = mdp.utils.mult(inv_xTx, self._xTy)

    def _execute(self, x):
        x = self._add_constant(x)
        return mdp.utils.mult(x, self.weights)

    def _add_constant(self, x):
        """Add a constant term to the vector 'x'.
        x -> [1 x]
        """
        return mdp.numx.concatenate((mdp.numx.ones((x.shape[0], 1),
                                           dtype=self.dtype), x), axis=1)
