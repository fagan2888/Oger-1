# RBM based nodes. So far this file only contains the CRBM and supports only
# binary units as it is based on the RBMNode that has the same limitation.
# TODO: fix the energy functions so that they are correct for gaussian units.

import Engine
import mdp
from mdp.utils import mult
from mdp.nodes import RBMNode

random = mdp.numx_rand.random
randn = mdp.numx_rand.randn
exp = mdp.numx.exp


class ERBMNode(RBMNode):
    """'Enhanced' Restricted Boltzmann Machine node. This node implements the
    same model as the RBMNode class but has some additional functionality.
    Gaussian units can be used, a method has been added that returns the
    contrastive divergence gradient and the node has been made invertible.

    For simplicity, the Gaussian units are assumed to have unit variance.
    """

    def __init__(self, visible_dim, hidden_dim, gaussian=False, dtype=None):
        """
        Arguments:

        hidden_dim -- number of hidden variables
        visible_dim -- number of observed variables
        gaussian -- use gaussian visible units (default is binary)
        """
        super(RBMNode, self).__init__(visible_dim, hidden_dim, dtype)
        self._initialized = False
        self._gaussian = gaussian


    def _energy(self, v, h):
        if self._gaussian:
            return ((((v - self.bv) ** 2).sum() / 2) - mult(h, self.bh) - 
                    (mult(v, self.w) * h).sum(axis=1))
        else:
            return (-mult(v, self.bv) - mult(h, self.bh) - 
                    (mult(v, self.w) * h).sum(axis=1))

    def train(self, v, n_updates=1, epsilon=0.1, decay=0., momentum=0.):
        """Update the internal structures according to the input data 'v'.
        The training is performed using Contrastive Divergence (CD).

        v -- a binary matrix having different variables on different columns
             and observations on the rows
        n_updates -- number of CD iterations. Default value: 1
        epsilon -- learning rate. Default value: 0.1
        decay -- weight decay term. Default value: 0.
        momentum -- momentum term. Default value: 0.
        """        

        if not self.is_training():
            errstr = "The training phase has already finished."
            raise mdp.TrainingFinishedException(errstr)

        self._train_phase_started = True

        self._train(v, n_updates, epsilon, decay, momentum)

    def _train(self, v, n_updates=1, epsilon=0.1, decay=0., momentum=0.):
        """Update the internal structures according to the input data 'v'.
        The training is performed using Contrastive Divergence (CD).

        v -- a binary matrix having different variables on different columns
             and observations on the rows
        n_updates -- number of CD iterations. Default value: 1
        epsilon -- learning rate. Default value: 0.1
        decay -- weight decay term. Default value: 0.
        momentum -- momentum term. Default value: 0.
        """        

        if not self._initialized:
            self._init_weights()

        # useful quantities
        n = v.shape[0]
        w, bv, bh = self.w, self.bv, self.bh

        # old gradients for momentum term
        dw, dbv, dbh = self._delta

        # get contrastive divergence gradient
        dwt, dbvt, dbht = self.get_CD_gradient(v, n_updates)

        # update parameters
        dw = momentum * dw + epsilon * dwt - decay * w
        w += dw

        dbv = momentum * dbv + epsilon * dbvt
        bv += dbv

        dbh = momentum * dbh + epsilon * dbht
        bh += dbh

        self._delta = (dw, dbv, dbh)


    def get_CD_gradient(self, v, n_updates=1):
        """Use Gibbs sampling to estimate the contrastive divergence gradient.

        v -- a binary matrix having different variables on different columns
             and observations on the rows
        n_updates -- number of CD iterations. Default value: 1

        Returns a tuple (dw, dbv, dbh) that contains the gradients of the
        weights and the biases of the visibles and the hidden respectively.
        """

        n = v.shape[0]

        # first update of the hidden units for the data term
        ph_data, h_data = self._sample_h(v)
        # n updates of both v and h for the model term
        h_model = h_data.copy()
        for i in range(n_updates):
            pv_model, v_model = self._sample_v(h_model)
            if self._gaussian:
                ph_model, h_model = self._sample_h(pv_model)
            else:
                ph_model, h_model = self._sample_h(v_model)
        
        # find dw
        data_term = mult(v.T, ph_data)
        model_term = mult(v_model.T, ph_model)
        dw = (data_term - model_term) / n
        
        # find dbv
        data_term = v.sum(axis=0)
        model_term = v_model.sum(axis=0)
        dbv = (data_term - model_term) / n

        # find dbh
        data_term = ph_data.sum(axis=0)
        model_term = ph_model.sum(axis=0)
        dbh = (data_term - model_term) / n

        return (dw, dbv, dbh)

    def is_invertible(self):
        return True

    def _sample_v(self, h):
        # returns  P(v=1|h,W,b) and a sample from it
        v_in = self.bv + mult(h, self.w.T)
        if self._gaussian:
            return v_in, v_in
        else:
            probs = 1. / (1. + exp(-v_in))
            v = (probs > random(probs.shape)).astype(self.dtype)
            return probs, v

    def _inverse(self, y, return_probs=True):
        """If 'return_probs' is True, returns the mean field of the
        visible variables v[n,i] conditioned on the states of the hiddens. 
        If 'return_probs' is False, return a sample from that probability.
        """
        probs, v = self._sample_v(y)
        if return_probs:
            return probs
        else:
            return v


class CRBMNode(ERBMNode):
    """Conditional Restricted Boltzmann Machine node. This type of
    RBM models the joint probability of the hidden and visible
    variables conditioned on a certain context variable. See the
    documentation of the RBMNode for more information.

    The context variables are expected to be concatendated to the
    input data. Note that the sample functions do however expect
    these types of variables as separated arguments. This has been
    done to allow for easier construction of flows while being
    able to specify context data on the fly as well.
    """

    def __init__(self, hidden_dim, visible_dim=None, context_dim=None,
                 gaussian=False, dtype=None):
        """
        Arguments:

        hidden_dim -- number of hidden variables
        visible_dim -- number of observed variables
        context_dim -- number of context variables
        gaussian -- use gaussian visible units (default is binary)
        """
        super(RBMNode, self).__init__(hidden_dim, visible_dim + context_dim, dtype)
        self._input_dim = visible_dim + context_dim
        self._output_dim = hidden_dim

        self.context_dim = context_dim
        self.visible_dim = visible_dim
        self._initialized = False

        self._gaussian = gaussian

    def _init_weights(self):
        # weights and biases are initialized to small random values to
        # break the symmetry that might lead to degenerate solutions during
        # learning
        self._initialized = True
        
        # undirected weights
        self.w = self._refcast(randn(self.visible_dim, self.output_dim) * 0.01)
        # context to visible weights
        self.a = self._refcast(randn(self.context_dim, self.visible_dim) * 0.01)
        # context to hidden weights
        self.b = self._refcast(randn(self.context_dim , self.output_dim) * 0.01)
        # bias on the visible (input) units
        self.bv = self._refcast(randn(self.visible_dim) * 0.01)
        # bias on the hidden (output) units
        self.bh = self._refcast(randn(self.output_dim) * 0.01)

        # delta w, a, b, bv, bh used for momentum term
        self._delta = (0., 0., 0., 0., 0.)

    def _split_data(self, x):
        # split data into visibles and context respectively.
        return x[:, :self.visible_dim], x[:, self.visible_dim:]

    def _sample_h(self, v, x):
        # returns P(h=1|v,W,b) and a sample from it
        dynamic_b = mult(x, self.b)
        probs = Engine.utils.LogisticFunction.f(self.bh + mult(v, self.w) + dynamic_b)
        h = (probs > random(probs.shape)).astype(self.dtype)
        return probs, h

    def _sample_v(self, h, x):
        # returns  P(v=1|h,W,b) and a sample from it
        dynamic_b = mult(x, self.a)
        v_in = self.bv + mult(h, self.w.T) + dynamic_b
        if self._gaussian:
            return v_in, v_in
        else:
            probs = Engine.utils.LogisticFunction.f(v_in)
            v = (probs > random(probs.shape)).astype(self.dtype)
            return probs, v

    def train(self, x, n_updates=1, epsilon=0.1, decay=0., momentum=0.):
        """Update the parameters according to the input 'v' and context 'x'.
        The training is performed using Contrastive Divergence (CD).

        v -- a binary matrix having different variables on different columns
             and observations on the rows.
        x -- a matrix having different variables on different columns and
             observations on the rows.
        n_updates -- number of CD iterations. Default value: 1
        epsilon -- learning rate. Default value: 0.1
        decay -- weight decay term. Default value: 0.
        momentum -- momentum term. Default value: 0.
        """

        if not self.is_training():
            errstr = "The training phase has already finished."
            raise mdp.TrainingFinishedException(errstr)

        #self._check_input(x)

        self._train_phase_started = True
        self._train(x, n_updates, epsilon, decay, momentum)

    def get_CD_gradient(self, x, n_updates=1):
        """Use Gibbs sampling to estimate the contrastive divergence gradient.

        x -- a binary matrix having different variables on different columns
             and observations on the rows (concatenation of visibles and context)
        n_updates -- number of CD iterations. Default value: 1

        Returns a tuple (dw, dbv, dbh, da, db) that contains the gradients of the
        weights and the biases of the visibles and the hidden respectively and
        the autoregressive gradients da and db.
        """

        # useful quantities
        n = x.shape[0]
        v, x = self._split_data(x)
        w, a, b, bv, bh = self.w, self.a, self.b, self.bv, self.bh

        # first update of the hidden units for the data term
        ph_data, h_data = self._sample_h(v, x)
        # n updates of both v and h for the model term
        h_model = h_data.copy()
        for i in range(n_updates):
            pv_model, v_model = self._sample_v(h_model, x)
            ph_model, h_model = self._sample_h(v_model, x)
        
        # find dw
        data_term = mult(v.T, ph_data)
        model_term = mult(v_model.T, ph_model)
        dw = (data_term - model_term) / n

        # find da
        data_term = v
        model_term = v_model
        # Should I include the weight decay here as well?
        da = mult(x.T, data_term - model_term) / n
        
        # find db
        data_term = ph_data
        model_term = ph_model
        db = mult(x.T, data_term - model_term) / n
        
        # find dbv
        data_term = v.sum(axis=0)
        model_term = v_model.sum(axis=0)
        dbv = (data_term - model_term) / n

        # find dbh
        data_term = ph_data.sum(axis=0)
        model_term = ph_model.sum(axis=0)
        dbh = (data_term - model_term) / n

        return (dw, dbv, dbh, da, db)

    def _train(self, x, n_updates=1, epsilon=0.1, decay=0., momentum=0.):
        """Update the parameters according to the input 'v' and context 'x'.
        The training is performed using Contrastive Divergence (CD).

        v -- a binary matrix having different variables on different columns
             and observations on the rows.
        x -- a matrix having different variables on different columns and
             observations on the rows.
        n_updates -- number of CD iterations. Default value: 1
        epsilon -- learning rate. Default value: 0.1
        decay -- weight decay term. Default value: 0.
        momentum -- momentum term. Default value: 0.
        """        
        if not self._initialized:
            self._init_weights()
 

        # useful quantities
        n = x.shape[0]
        w, a, b, bv, bh = self.w, self.a, self.b, self.bv, self.bh

        # old gradients for momentum term
        dw, da, db, dbv, dbh = self._delta
        
        # get the gradient
        dwt, dbvt, dbht, dat, dbt = self.get_CD_gradient(x, n_updates)

        # update w
        dw = momentum * dw + epsilon * dwt - decay * w
        w += dw

        # update a
        da = momentum * da + epsilon * dat - decay * a
        a += da
        
        # update b
        db = momentum * db + epsilon * dbt - decay * b
        b += db
        
        # update bv
        dbv = momentum * dbv + epsilon * dbvt
        bv += dbv

        # update bh
        dbh = momentum * dbh + epsilon * dbht
        bh += dbh

        self._delta = (dw, da, db, dbv, dbh)


    def sample_h(self, v, x):
        """Sample the hidden variables given observations v and context x.

        Returns a tuple (prob_h, h), where prob_h[n,i] is the
        probability that variable 'i' is one given the observations
        v[n,:], and h[n,i] is a sample from the posterior probability."""

        # The pre execution checks assume that v will give the input_dim but
        # this is not correct anymore due to the concatenation for execute.
        # Perhaps I should make two versions of the sample functions. One type
        # for the merged and one type for the separated input variables.
        #self._pre_execution_checks(v)
        return self._sample_h(v, x)

    def sample_v(self, h, x):
        """Sample the observed variables given hidden variables h and context.

        Returns a tuple (prob_v, v), where prob_v[n,i] is the
        probability that variable 'i' is one given the hidden variables
        h[n,:], and v[n,i] is a sample from that conditional probability."""

        #self._pre_inversion_checks(h)
        return self._sample_v(h, x)

    def _energy(self, v, h, x):
        ba = mult(x, self.a)
        bb = mult(x, self.b)
        ba += self.bv
        bb += self.bh
        if self._gaussian:
            return (((v - ba) ** 2).sum() / 2 - (h * bb).sum(axis=1) - 
                    (mult(v, self.w) * h).sum(axis=1))
        else:
            return (-(v * ba).sum(axis=1) - (h * bb).sum(axis=1) - 
                    (mult(v, self.w) * h).sum(axis=1))

    def energy(self, v, h, x):
        """Compute the energy of the RBM given observed variables state 'v' and
        hidden variables state 'h'."""
        return self._energy(v, h, x)

    def _execute(self, x, return_probs=True):
        """If 'return_probs' is True, returns the probability of the
        hidden variables h[n,i] being 1 given the observations v[n,:] and
        the context state x.
        If 'return_probs' is False, return a sample from that probability.
        """
        v, x = self._split_data(x)
        probs, h = self._sample_h(v, x)
        if return_probs:
            return probs
        else:
            return h

