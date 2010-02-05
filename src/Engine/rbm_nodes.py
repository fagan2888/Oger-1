# RBM based nodes. So far this file only contains the CRBM and supports only
# binary units as it is based on the RBMNode that has the same limitation.
import mdp
from Engine.utility_functions import logistic
from mdp.utils import mult
from mdp.nodes import RBMNode

random = mdp.numx_rand.random
randn = mdp.numx_rand.randn
exp = mdp.numx.exp

class CRBMNode(RBMNode):
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

    def __init__(self, hidden_dim, visible_dim=None, context_dim=None, dtype=None):
        """
        Arguments:

        hidden_dim -- number of hidden variables
        visible_dim -- number of observed variables
        context_dim -- number of context variables
        """
        super(RBMNode, self).__init__(hidden_dim, visible_dim + context_dim, dtype)
        self._input_dim = visible_dim + context_dim
        self._output_dim = hidden_dim

        self.context_dim = context_dim
        self.visible_dim = visible_dim
        self._initialized = False

    def _init_weights(self):
        # weights and biases are initialized to small random values to
        # break the symmetry that might lead to degenerate solutions during
        # learning
        self._initialized = True
        
        # undirected weights
        self.w = self._refcast(randn(self.visible_dim, self.output_dim)*0.01)
        # context to visible weights
        self.a = self._refcast(randn(self.context_dim, self.visible_dim)*0.01)
        # context to hidden weights
        self.b = self._refcast(randn(self.context_dim , self.output_dim)*0.01)
        # bias on the visible (input) units
        self.bv = self._refcast(randn(self.visible_dim)*0.01)
        # bias on the hidden (output) units
        self.bh = self._refcast(randn(self.output_dim)*0.01)

        # delta w, a, b, bv, bh used for momentum term
        self._delta = (0., 0., 0., 0., 0.)

    def _split_data(self, x):
        # split data into visibles and context respectively.
        return x[:, :self.visible_dim], x[:, self.visible_dim:]

    def _sample_h(self, v, x):
        # returns P(h=1|v,W,b) and a sample from it
        dynamic_b = mult(x, self.b)
        probs = logistic(self.bh + mult(v, self.w) + dynamic_b)
        h = (probs > random(probs.shape)).astype(self.dtype)
        return probs, h

    def _sample_v(self, h, x):
        # returns  P(v=1|h,W,b) and a sample from it
        dynamic_b = mult(x, self.a)
        probs = logistic(self.bv + mult(h, self.w.T) + dynamic_b)
        v = (probs > random(probs.shape)).astype(self.dtype)
        return probs, v

    def train(self, x, n_updates=1, epsilon=0.1, decay=0., momentum=0.,
              verbose=False):
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
        self._train(x, n_updates, epsilon, decay, momentum, verbose)

    def _train(self, x, n_updates=1, epsilon=0.1, decay=0., momentum=0.,
               verbose=False):
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
 
        v, x = self._split_data(x)

        # useful quantities
        n = v.shape[0]
        w, a, b, bv, bh = self.w, self.a, self.b, self.bv, self.bh

        # old gradients for momentum term
        dw, da, db, dbv, dbh = self._delta
        
        # first update of the hidden units for the data term
        ph_data, h_data = self._sample_h(v, x)
        # n updates of both v and h for the model term
        h_model = h_data.copy()
        for i in range(n_updates):
            pv_model, v_model = self._sample_v(h_model, x)
            ph_model, h_model = self._sample_h(v_model, x)
        
        # update w
        data_term = mult(v.T, ph_data)
        model_term = mult(v_model.T, ph_model)
        dw = momentum*dw + epsilon*((data_term - model_term)/n - decay*w)
        w += dw

        # update a
        data_term = v
        model_term = v_model
        # Should I include the weight decay here as well?
        da = momentum*da + epsilon*((mult(x.T, data_term - model_term))/n)
        a += da
        
        # update b
        data_term = ph_data
        model_term = ph_model
        db = momentum*db + epsilon*((mult(x.T, data_term - model_term))/n)
        b += db
        
        # update bv
        data_term = v.sum(axis=0)
        model_term = v_model.sum(axis=0)
        dbv = momentum*dbv + epsilon*((data_term - model_term)/n)
        bv += dbv

        # update bh
        data_term = ph_data.sum(axis=0)
        model_term = ph_model.sum(axis=0)
        dbh = momentum*dbh + epsilon*((data_term - model_term)/n)
        bh += dbh

        self._delta = (dw, da, db, dbv, dbh)
        self._train_err = float(((v-v_model)**2.).sum())

        if verbose:
            print 'training error', self._train_err/v.shape[0]
            ph, h = self._sample_h(v, x)
            print 'energy', self._energy(v, ph, x).sum()

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
        # TODO Check why the energy keeps climbing with a high learning rate.
        # Perhaps the autoregressive weights should have a lower learning rate
        # than the other weights.
        ba = mult(x, self.a)
        bb = mult(x, self.b)
        ba += self.bv
        bb += self.bh
        return -(v * ba).sum(axis=1) - (h * bb).sum(axis=1) - (mult(v, self.w)*h).sum(axis=1)

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

