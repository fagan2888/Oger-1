"""
This subpackage contains a number of Oger-related nodes. It contains several additional MDP nodes such as RBM nodes, reservoir nodes and signal processing nodes.
"""

from flows import InspectableFlow
from reservoir_nodes import (ReservoirNode, LeakyReservoirNode, TrainableReservoirNode, HebbReservoirNode, FeedbackReservoirNode, BandpassReservoirNode, CUDAReservoirNode)
from linear_nodes import (RidgeRegressionNode, ParallelLinearRegressionNode)
from nonlinear_nodes import (ThresholdNode, PerceptronNode)
from rbm_nodes import (ERBMNode, CRBMNode, CUDACRBMNode, CUDATRMNode)
from utility_nodes import (FeedbackNode, MeanAcrossTimeNode, WTANode, ShiftNode, ResampleNode, TimeFramesNode2)
from spiking_nodes import (BrianIFReservoirNode, SpikingIFReservoirNode)
#from ode_nodes import (OdeNode)

# clean up namespace
del flows
del reservoir_nodes
del rbm_nodes 
del linear_nodes
del nonlinear_nodes
del utility_nodes
del spiking_nodes

__all__ = ['InspectableFlow', 'ReservoirNode', 'LeakyReservoirNode', 'TrainableReservoirNode', 'HebbReservoirNode', 'FeedbackReservoirNode', 'RidgeRegressionNode', 'ParallelLinearRegressionNode', 'ThresholdNode', 'PerceptronNode', 'ERBMNode', 'CRBMNode', 'CUDATRMNode', 'FeedbackNode', 'WashoutNode', 'MeanAcrossTimeNode', 'WTANode', 'ShiftNode', 'ResampleNode', 'TimeFramesNode2', 'SpikingIFReservoirNode']
