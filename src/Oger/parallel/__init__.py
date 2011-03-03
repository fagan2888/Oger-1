"""
This subpackage contains support for parallel processing on a computing grid. 
"""

from parallel import (GridScheduler)

# clean up namespace
del parallel
__all__ = ['GridScheduler']
