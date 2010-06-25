'''
    This example shows a module that is being developed for processing spatial data (such as handwriting) more efficiently. 
    It is inspired by convolutional neural networks, and employs many weight-sharing reservoirs that receive input from local 
    receptive fields. The model is trained in an unsupervised way to extract slowly varying spatial features in the input using 
    a similar approach to Slow Feature Analysis. Multiple such modules can be stacked to form a hierarchical architecture 
    for spatial data processing.

    @author: Vytenis Sakenas
'''

import mdp
import Engine
import pylab
import scipy.io
 
class SubtractNode(mdp.Node):
    '''
        Subtracts last input_dim/2 dimensions of data from first input_dim/2 dimensions of data
    '''
    def __init__(self, input_dim=2, output_dim=None):
        super(SubtractNode,self).__init__(input_dim=input_dim, output_dim=output_dim)
        if (input_dim%2 != 0):
            raise mdp.NodeException()
        if self.output_dim == None:
            self.output_dim = self.input_dim/2
        if self.output_dim != self.input_dim/2:
            raise mdp.NodeException()
        
    def is_trainable(self):
        return False
    
    def is_invertible(self):
        return False

    def _execute(self, x):
        return x[:,0:self.output_dim] - x[:,self.output_dim:]
    

class UntrainableFlowNode(mdp.hinet.FlowNode):
    '''
        Node to avoid train on execute MDP behavior
    '''
    def is_trainable(self):
        return False   

def CreateConvolutionalMapping(input_features, input_size, lrf_size, lrf_shift):
    """ 
        Creates a connection array for a Switchboard which implements local receptive fields
            input_features - the number of features at a single position that is fed
            input_size - the number of dimensions in the input data
            lrf_size - number of dimensions to feed to a single processing node
            lrf_shift - shift in input dimension for the following feeding point to a processing node
    """
    assert input_size>=lrf_size
    
    n_units = (input_size-lrf_size)/lrf_shift + 1
    
    conn = mdp.numx.arange(n_units*lrf_size*input_features, dtype=int)
    for i in range(n_units):
        for j in range(lrf_size*input_features):
            conn[i*lrf_size*input_features+j] = i*lrf_shift*input_features+j;
    return conn

def CreateSubtractMapping(reservoir_size, n_units):
    ''' 
        Creates a connection array for a Switchboard routing the data to SubtractNode
            reservoir_size - the size of a single reservoir
            n_units - the total number of reservoirs 
    '''
    conns = mdp.numx.arange((n_units-1)*2*reservoir_size, dtype=int)
    for i in range(n_units-1):
        for j in range(2*reservoir_size):
            conns[i*2*reservoir_size+j] = i*reservoir_size+j
    return conns

def ConstructSpatialProcessingLayer(lrf_size = 6, lrf_shift = 1, input_features = 1, input_size = 12, reservoir_size = 20, out_features = 10):
    '''
        Builds the module for spatial data processing
        
            lrf_size - a number of input dimensions a single reservoir is connected
            lrf_shift - a number of input dimensions to shift between the input fields of the adjacent reservoirs
            input_features - number of features in the input data
            input_size - the vertical size of the input
            reservoir_size - size of the reservoir that will be used in the module
            out_features - number of slowest features to extract
        
        Returns a tuple: a flow to be used for training and a flow to be used in exploitation phase
    '''
    
    n_units = (input_size-lrf_size)/lrf_shift + 1       # Number of reservoirs
    
    # Local receptive fields
    sboard = mdp.hinet.Switchboard(input_dim=12, connections=CreateConvolutionalMapping(input_features,input_size,lrf_size,lrf_shift))
    # The reservoir is followed by a whitening node and put into a CloneLayer to achieve weight sharing    
    reservoir = Engine.nodes.ReservoirNode(input_dim=lrf_size*input_features, output_dim=reservoir_size, input_scaling=0.5)
    white_node = mdp.nodes.WhiteningNode(input_dim=reservoir_size, output_dim=reservoir_size)
    in_flow_node = mdp.hinet.FlowNode(mdp.Flow([reservoir, white_node]))
    reservoir_layer = mdp.hinet.CloneLayer(in_flow_node, n_nodes=n_units)
    
    # Route output to SubtractNode
    sboard2 = mdp.hinet.Switchboard(input_dim=n_units*reservoir_size, connections=CreateSubtractMapping(reservoir_size, n_units))
    # Subtract outputs of the adjacent reservoirs
    snode = SubtractNode(input_dim=2*reservoir_size)
    # Compute PCA of the differences; slowly spatialy changing features will be in directions of
    # principal components with smallest values
    pcanode = mdp.nodes.PCANode(input_dim=reservoir_size, output_dim=reservoir_size)
    in_flow_node2 = mdp.hinet.FlowNode(mdp.Flow([snode, pcanode]))
    fe_layer = mdp.hinet.CloneLayer(in_flow_node2, n_nodes=n_units-1)
    
    # Connect eveything into a single flow for training
    train_flow = mdp.Flow([sboard,reservoir_layer,sboard2,fe_layer])
    
    # For testing create a different flow, because PCANode is used in different place
    test_in_flow_node = UntrainableFlowNode(mdp.Flow([reservoir, white_node, pcanode]))
    test_reservoir_layer = mdp.hinet.CloneLayer(test_in_flow_node, n_nodes=n_units)
    test_flow = mdp.Flow([sboard, test_reservoir_layer])
    
    return (train_flow,test_flow)



if __name__ == '__main__':
    data = scipy.io.loadmat('../datasets/vytenis_printedChars.mat')    # Printed chars with noise data
    u = data['data']                                            # Training data
    visdata = data['data'][:,0:300]                             # Data for visualization
    
    res_size = 20                                               # Size of the reservoir
    
    # Create two flows: one for training and one for testing
    (trainf, testf) = ConstructSpatialProcessingLayer(lrf_size = 5, lrf_shift = 1, input_features = 1, input_size = 12, reservoir_size = res_size, out_features = 10)
    
    trainf.train(u.T)                                           # Do training
    
    out = testf.execute(visdata.T)                              # Run with visualization data
    
    # Plot data and 5 spatialy slowest features 
    pylab.subplot(6,1,1)
    pylab.imshow(visdata,aspect='auto')
    for i in range(5):
        pylab.subplot(6,1,i+2)
        pylab.imshow(out[:,res_size-5+i::res_size].T,aspect='auto',interpolation='nearest')
    pylab.show()