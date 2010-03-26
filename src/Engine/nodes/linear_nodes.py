import mdp

class RidgeRegressionNode(mdp.nodes.LinearRegressionNode):
    """Ridge Regression node. Extends the LinearRegressionNode and adds an additional
    ridge_param parameter.
    
    Solves the following equation: (AA^T+\lambdaI)^-(1)A^TB
    """
    def __init__(self, ridge_param=0, with_bias=True, use_pinv=False, input_dim=None, output_dim=None, dtype=None):
        super(RidgeRegressionNode, self).__init__(input_dim=input_dim, output_dim=output_dim, with_bias=with_bias, use_pinv=use_pinv, dtype=dtype)
        
        self.ridge_param = ridge_param

    def _stop_training(self):
        try:
            if self.use_pinv:
                invfun = mdp.utils.pinv
            else:
                invfun = mdp.utils.inv
            inv_xTx = invfun(self._xTx + self.ridge_param * mdp.numx.eye(self._input_dim+1))
        except mdp.numx_linalg.LinAlgError, exception:
            errstr = (str(exception) + 
                      "\n Input data may be redundant (i.e., some of the " +
                      "variables may be linearly dependent).")
            raise mdp.NodeException(errstr)
        
        print "training"
        self.beta = mdp.utils.mult(inv_xTx, self._xTy)

class ParallelLinearRegressionNode(mdp.parallel.ParallelExtensionNode, mdp.nodes.LinearRegressionNode):
    """Parallel extension for the LinearRegressionNode and all its derived classes
    (eg. RidgeRegressionNode).
    """
    def _fork(self):
        return self._default_fork()
    
    def _join(self, forked_node):
        if self._xTx is None:
            self._xTx = forked_node._xTx
            self._xTy = forked_node._xTy
            self._tlen = forked_node._tlen
        else:
            self._xTx += forked_node._xTx
            self._xTy += forked_node._xTy
            self._tlen += forked_node._tlen
