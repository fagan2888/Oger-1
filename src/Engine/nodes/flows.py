import mdp
from mdp import numx

class InspectableFlow(mdp.Flow):
    """A flow that allows to inspect the last outputs of all the nodes in the flow.
    """
    def __init__(self, flow, crash_recovery=False, verbose=False):
        super(InspectableFlow, self).__init__(flow, crash_recovery, verbose)
        
        self._states = [[] for _ in range(len(self.flow))]
                
    def _execute_seq(self, x, nodenr = None):
        """This code is a copy from the code from mdp.Flow, but with additional
        state tracking.
        """
        flow = self.flow
        if nodenr is None:
            nodenr = len(flow)-1
        for i in range(nodenr+1):
            try:
                x = flow[i].execute(x)                    
                self._states[i].append(x)
            except Exception, e:
                self._propagate_exception(e, i)
        return x

    def execute(self, iterable, nodenr = None):
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
        for i in range(len(flow)-1, -1, -1):
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