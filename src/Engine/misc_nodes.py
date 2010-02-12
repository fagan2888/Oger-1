import mdp
from mdp import Node


class ShiftNode(Node):
    """Return input data shifted one or more time steps.

    This is useful for architectures in which data from different time steps is
    needed. The values that are left over are set to zero.

    Negative shift values cause a shift to the left and positive ones to the
    right.
    """

    def __init__(self, input_dim=None, output_dim=None, n_shifts=1,
                 dtype='float64'):
        super(ShiftNode, self).__init__(input_dim, output_dim, dtype)
        self.n_shifts = n_shifts

    def is_trainable(self):
        False

    def _execute(self, x):
        n, d = x.shape
        assert(n > 1)

        ns = self.n_shifts
        y = x.copy()

        if ns < 0:
            y[:ns] = x[-ns:]
            y[ns:] = 0
        elif ns > 0:
            y[ns:] = x[:-ns]
            y[:ns] = 0
        else:
            y = x

        return y

    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = n

