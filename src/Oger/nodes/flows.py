import mdp
from mdp import numx

# TODO: biMDP has the ability to introspect flows and nodes, we could use
# that mechanism instead of this custom flow. This would possibly also
# allow parallelization 
class InspectableFlow(mdp.Flow):
    """A flow that allows to inspect the last outputs of all the nodes in the flow.
    """
    def __init__(self, flow, crash_recovery=False, verbose=False):
        super(InspectableFlow, self).__init__(flow, crash_recovery, verbose)

        self._states = [[] for _ in range(len(self.flow))]

    def _execute_seq(self, x, nodenr=None):
        """This code is a copy from the code from mdp.Flow, but with additional
        state tracking.
        """
        flow = self.flow
        if nodenr is None:
            nodenr = len(flow) - 1
        for i in range(nodenr + 1):
            try:
                x = flow[i].execute(x)
                self._states[i].append(x)
            except Exception, e:
                self._propagate_exception(e, i)
        return x

    def execute(self, iterable, nodenr=None):
        self._states = [[] for _ in range(len(self.flow))]

        output = super(InspectableFlow, self).execute(iterable, nodenr)

        for i in range(len(self.flow)):
            self._states[i] = numx.concatenate(self._states[i])

        return output

    def _inverse_seq(self, x):
        """This code is a copy from the code from mdp.Flow, but with additional
        state tracking.
        """
        flow = self.flow
        for i in range(len(flow) - 1, -1, -1):
            try:
                x = flow[i].inverse(x)
                self._states[i].append(x)
            except Exception, e:
                self._propagate_exception(e, i)
        return x

    def inverse(self, iterable):
        self._states = [list() for _ in range(len(self.flow))]

        output = super(InspectableFlow, self).inverse(iterable)

        for i in range(len(self.flow)):
            self._states[i] = numx.concatenate(self._states[i])

        return output

    def inspect(self, node_or_nr):
        """Return the state of the given node or node number in the flow.
        """
        if isinstance(node_or_nr, mdp.Node):
            return self._states[self.flow.index(node_or_nr)]
        else:
            return self._states[node_or_nr]

class FreerunFlow(mdp.Flow):
    def __init__(self, flow, crash_recovery=False, verbose=False, freerun_steps=None):
        super(FreerunFlow, self).__init__(flow, crash_recovery, verbose)
        if freerun_steps is None:
            errstr = ("The FreerunFlow must be initialized with an explicit freerun horizon.")
            raise mdp.FlowException(errstr)
        self.freerun_steps = freerun_steps

    def train(self, data_iterables):

        data_iterables = self._train_check_iterables(data_iterables)

        # train each Node successively
        for i in range(len(self.flow)):
            if self.verbose:
                print "Training node #%d (%s)" % (i, str(self.flow[i]))
            if not data_iterables[i] == []:
                datax = [x[0:-1, :] for x in data_iterables[i][0]]
                datay = [x[1:, :] for x in data_iterables[i][0]]
            else:
                datax, datay = [], []
            self._train_node(zip(datax, datay), i)
            if self.verbose:
                print "Training finished"

        self._close_last_node()


    def execute(self, x, nodenr=None):
        if not isinstance(x, mdp.numx.ndarray):
            errstr = ("FreerunFlows can only be executed using numpy arrays as input.")
            raise mdp.FlowException(errstr)

        if self.freerun_steps >= x.shape[0]:
            errstr = ("Number of freerun steps (%d) should be less than the number of timesteps in x (%d)" % (self.freerun_steps, x.shape[0]))
            raise mdp.FlowException(errstr)


        # Run the flow for warmup
        if self.freerun_steps > x.shape[0]:
            errstr = ("The number of freerun steps (%d) is larger than the input (%d):" % (self.freerun_steps, x.shape[0]))
            raise mdp.FlowException(errstr)

        self._execute_seq(x[:-self.freerun_steps, :])
        self.fb_value = x[-self.freerun_steps, :]

        res = mdp.numx.zeros((self.freerun_steps, 1))
        for step in range(self.freerun_steps):
            res[step] = self.fb_value
            self.fb_value = self._execute_seq(mdp.numx.atleast_2d(self.fb_value))
        return mdp.numx.concatenate((x[:-self.freerun_steps, :], res))
