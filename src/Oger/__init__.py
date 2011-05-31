'''
Oger is a Python toolbox for rapidly building, training and evaluating modular learning architectures on large datasets. It builds functionality on top of the Modular toolkit for Data Processing (MDP).  Oger builds functionality on top of MDP, such as:
 - Cross-validation of datasets
 - Grid-searching large parameter spaces
 - Processing of temporal datasets
 - Gradient-based training of deep learning architectures
 - Interface to the Speech Processing, Recognition, and Automatic Annotation Kit (SPRAAK) 

In addition, several additional MDP nodes are provided by Oger, such as a:
 - Reservoir node
 - Leaky reservoir node
 - Ridge regression node
 - Conditional Restricted Boltzmann Machine (CRBM) node
'''

import utils
import nodes
import datasets
import gradient
import evaluation
#print 'Warning: importing parallel subpackage - still in development!'
#import parallel


__all__ = ['utils', 'nodes', 'datasets', 'gradient', 'evaluation']
