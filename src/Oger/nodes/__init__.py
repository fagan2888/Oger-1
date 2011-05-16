"""
This subpackage contains a number of Oger-related nodes. It contains several additional MDP nodes such as RBM nodes, reservoir nodes and signal processing nodes.
"""

from flows import (InspectableFlow, FreerunFlow)
from reservoir_nodes import (ReservoirNode, LeakyReservoirNode, TrainableReservoirNode, HebbReservoirNode, BandpassReservoirNode, CUDAReservoirNode)
from linear_nodes import (RidgeRegressionNode, ParallelLinearRegressionNode)
from nonlinear_nodes import (ThresholdNode, PerceptronNode, IRLSLogisticRegressionNode, LogisticRegressionNode)
from rbm_nodes import (ERBMNode, CRBMNode, CUDACRBMNode, CUDATRMNode)
from utility_nodes import (FeedbackNode, MeanAcrossTimeNode, WTANode, ShiftNode, FeedbackShiftNode, ResampleNode, TimeFramesNode2, RescaleZMUSNode, SupervisedLayer, MaxVotingNode)
try:
    from spiking_nodes import (BrianIFReservoirNode, SpikingIFReservoirNode)
    del spiking_nodes
except:
    pass
from layers import (SplitOutputLayer, SplitOutputSameInputLayer)
#from ode_nodes import (OdeNode)

# clean up namespace
del flows
del reservoir_nodes
del rbm_nodes
del linear_nodes
del nonlinear_nodes
del utility_nodes


__all__ = ['InspectableFlow', 'ReservoirNode', 'LeakyReservoirNode', 'TrainableReservoirNode', 'HebbReservoirNode',
           'RidgeRegressionNode', 'ParallelLinearRegressionNode', 'ThresholdNode', 'PerceptronNode', 'IRLSLogisticRegressionNode', 'LogisticRegressionNode',
           'ERBMNode', 'CRBMNode', 'CUDACRBMNode', 'CUDAReservoirNode', 'CUDATRMNode',
           'FeedbackNode', 'WashoutNode', 'MeanAcrossTimeNode', 'WTANode', 'ShiftNode', 'FeedbackShiftNode', 'ResampleNode', 'RescaleZMUSNode', 'SupervisedLayer', 'MaxVotingNode',
           'TimeFramesNode2', 'SpikingIFReservoirNode']
