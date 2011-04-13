"""
This subpackage contains support for parallel processing on a computing grid. 
"""

from parallel import (GridScheduler, ParallelFlow)
import parallel_optimization

# clean up namespace
del parallel
del parallel_optimization
__all__ = ['GridScheduler', 'ParallelFlow']
