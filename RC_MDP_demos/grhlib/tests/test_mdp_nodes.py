#===============================================================================
# Test scripts of self-made nodes for the MDP package
#===============================================================================

import sys
from numpy.testing import *
import numpy as np
import aureservoir as au
import mdp
sys.path.append("../")
from reservoir_nodes import ReservoirNode, LinearReadoutNode, IdentityNode
from reservoir_nodes import StateCompressionNode, VoteAverageNode, ReservoirArrayStateComprNode
from reservoir_nodes import SquareStatesNode

def plot_mdp_network(flow, htmlfile="test.html"):
    """ Generates a HTML graphic of the architecture.
    """
    file = open(htmlfile, "w")
    file.write('<html>\n<head>\n<title>HiNetHTML Test</title>\n</head>\n<body>\n')
    hinet_html = mdp.hinet.HiNetHTML(file)
    hinet_html.parse_flow(flow)
    file.write('</body>\n</html>')
    file.close()


class DataIterator:
    """ Iterator to train a network with multiple data chunks.
    """
    def __init__(self, data, sup=1):
        self.data = data
        self.sup = sup
    def __iter__(self):
        for chunk in self.data:
            if self.sup: yield chunk
            else: yield chunk[0]


class test_aureservoir(NumpyTestCase):
    """ Tests reservoir_nodes against the aureservoir implementation.
    """

    def setUp(self):
        """ Setup routine for all tests.
        """
        self.size = 10
        self.inputs = 2
        self.outputs = 2
        self.train_steps = 50 # durchh 5 teilbar !
        self.test_steps = 50
        self.washout = 8
        
        # construct aureservoir ESN
        self.net = au.DoubleESN()
        self.net.setInputs(self.inputs)
        self.net.setOutputs(self.outputs)
        self.net.setSize(self.size)

        # make some training and testing data
        self.train_in = np.random.rand(self.train_steps,self.inputs)
        self.test_in = np.random.rand(self.test_steps,self.inputs)
        self.train_out = np.random.rand(self.train_steps,self.outputs)
            
    def testSimpleESN(self, level=1):
        """ Test MDP-nodes against an ESN from aureservoir.
        """
        # construct mdp ESN
        reservoir = ReservoirNode(self.inputs, self.size,
                                  dtype='float64', prototype=self.net)
        readout = LinearReadoutNode(self.size+self.inputs, self.outputs,
                                    ignore=self.washout)

        # init networks and copy it to really have the same one
        self.net.init()
        reservoir.reservoir  = au.DoubleESN(self.net)

        # build hierarchical mdp network
        res = mdp.hinet.SameInputLayer([reservoir,
                                        IdentityNode(self.inputs, self.inputs)])
        mdp_net = mdp.Flow([res, readout])

        # train aureservoir ESN
        self.net.train(self.train_in.T.copy(), self.train_out.T.copy(),
                       self.washout)

        # train mdp network
        # (nodes which can't be trained can be given a None)
        mdp_net.train([None, [(self.train_in, self.train_out)]])

        # test for output weights
        au_w = self.net.getWout().copy().T
        mdp_w = mdp_net[1].W  # readout.W
        assert_array_almost_equal(au_w, mdp_w)

        # run aureservoir model with test data
        au_out = np.zeros((self.outputs, self.test_steps))
        self.net.simulate(self.test_in.T.copy(), au_out)

        # run mdp model with test data
        mdp_out = mdp_net(self.test_in)

        # compare output data
        assert_array_almost_equal(au_out.T, mdp_out)

    def testMultipleTrainCalls(self, level=1):
        """ Test multiple calls of train-method
        """
        # construct mdp ESN
        reservoir = ReservoirNode(self.inputs, self.size,
                                  dtype='float64', prototype=self.net)
        readout = LinearReadoutNode(self.size+self.inputs, self.outputs,
                                    ignore=self.washout)

        # init networks and copy it to really have the same one
        self.net.init()
        reservoir.reservoir  = au.DoubleESN(self.net)

        # build hierarchical mdp network
        res = mdp.hinet.SameInputLayer([reservoir,
                                        IdentityNode(self.inputs, self.inputs)])
        flow = mdp.Flow([res, readout])
        mdp_net = mdp.hinet.FlowNode(flow)

        # train aureservoir ESN
        self.net.train(self.train_in.T.copy(), self.train_out.T.copy(),
                       self.washout)

        # train mdp ESN in multiple stages
        stages = 5
        ssteps = self.train_steps / stages
        for n in range(stages):
            iin = self.train_in[n*ssteps:(n+1)*ssteps,:]
            oout = self.train_out[n*ssteps:(n+1)*ssteps,:]
            mdp_net.train(iin,oout)
            #mdp_net.train([None, None, [(iin,oout)]])
        mdp_net.stop_training()
        
        # test for output weights
        au_w = self.net.getWout().copy().T
        mdp_w = mdp_net._flow[1].W  # readout.W
        assert_array_almost_equal(au_w, mdp_w)

        # run aureservoir model with test data
        au_out = np.zeros((self.outputs, self.test_steps))
        self.net.simulate(self.test_in.T.copy(), au_out)

        # run mdp model with test data
        mdp_out = mdp_net(self.test_in)

        # compare output data
        assert_array_almost_equal(au_out.T, mdp_out)

    def testWithIterator(self, level=1):
        """ Test MDP-nodes with iterator training and multiple calls
        """
        # construct mdp ESN
        reservoir = ReservoirNode(self.inputs, self.size,
                                  dtype='float64', prototype=self.net)
        readout = LinearReadoutNode(self.size+self.inputs, self.outputs,
                                    ignore=self.washout)

        # init networks and copy it to really have the same one
        self.net.init()
        reservoir.reservoir  = au.DoubleESN(self.net)

        # build hierarchical mdp network
        res = mdp.hinet.SameInputLayer([reservoir,
                                        IdentityNode(self.inputs, self.inputs)])
        mdp_net = mdp.Flow([res, readout])

        # train aureservoir ESN
        self.net.train(self.train_in.T.copy(), self.train_out.T.copy(),
                       self.washout)

        # train mdp ESN with iterator, 1 stage
#        traindata = (self.train_in, self.train_out)
#        mdp_net.train([None,
#                       None,
#                       DataIterator(traindata)] )
        
        # train mdp ESN in multiple stages
        stages = 5
        ssteps = self.train_steps / stages
        traindata = []
        for n in range(stages):
            chunk = (self.train_in[n*ssteps:(n+1)*ssteps,:],
                     self.train_out[n*ssteps:(n+1)*ssteps,:])
            traindata.append(chunk)
            
        mdp_net.train([None, DataIterator(traindata)] )

        # test for output weights
        au_w = self.net.getWout().copy().T
        mdp_w = mdp_net[1].W  # readout.W
        assert_array_almost_equal(au_w, mdp_w)

        # run aureservoir model with test data
        au_out = np.zeros((self.outputs, self.test_steps))
        self.net.simulate(self.test_in.T.copy(), au_out)

        # run mdp model with test data
        mdp_out = mdp_net(self.test_in)

        # compare output data
        assert_array_almost_equal(au_out.T, mdp_out)

    def testStateCompression(self, level=1):
        """ Test a reservoir with state compression in the readout.
        
        (Not really a test, should just show how to build up these networks ...)
        """
        compr_points = 5
        
        # construct mdp ESN
        reservoir = ReservoirNode(self.inputs, self.size,
                                  dtype='float64', prototype=self.net)
        compr = StateCompressionNode(self.size+self.inputs,
                                     support_points=compr_points)
        readout = LinearReadoutNode(compr_points*(self.size+self.inputs),
                                    self.outputs, ignore=0, use_pi=1)

        # build hierarchical mdp network
        res = mdp.hinet.SameInputLayer([reservoir,
                               IdentityNode(self.inputs, self.inputs)])
        flow = mdp.Flow([res, compr, readout])
#        plot_mdp_network(flow,"mdp_network_1.html")
        mdp_net = mdp.hinet.FlowNode(flow)
        
        # train mdp network
        trainout = self.train_out[-1,:].reshape(1,-1)
        mdp_net.train(self.train_in, trainout)
        mdp_net.stop_training()
#        mdp_net.train([None, None, [(self.train_in, trainout)]])

        # run the model with test data and check the shape
        mdp_out = mdp_net(self.test_in)
        assert mdp_out.shape == trainout.shape, 'incorrect output shape'

    def testStateCompressionArray(self, level=1):
        """ Test an array of reservoirs with state compression in the readout.
        
        Only checks if the reservoir have the right shape and that all weight
        matrices are different.
        """
        compr_points = 5
        nr_experts = 2  # nr of parallel reservoirs
        
        mdp_net = ReservoirArrayStateComprNode(self.inputs, self.outputs, 'float64',
                                               nr_experts, compr_points, self.net)
        
        # train mdp network
        trainout = self.train_out[-1,:].reshape(1,-1)
        mdp_net.train(self.train_in, trainout)
        mdp_net.stop_training()
        
        # run the model with test data and check the shape
        mdp_out = mdp_net(self.test_in)
        assert mdp_out.shape == trainout.shape, 'incorrect output shape'
        
        # check if the two reservoirs are not the same
        win1 = mdp_net.network._flow[0].nodes[0]._flow[0].nodes[0].reservoir.getWin()
        win2 = mdp_net.network._flow[0].nodes[1]._flow[0].nodes[0].reservoir.getWin()
        assert win1[0,0] != win2[0,0], 'input weights are the same'
        assert win1[0,1] != win2[0,1], 'input weights are the same'
        assert win1[5,0] != win2[5,0], 'input weights are the same'
        assert win1[5,1] != win2[5,1], 'input weights are the same'


class test_nodes(NumpyTestCase):
    """ Tests the individual nodes.
    """

    def setUp(self):
        """ Setup routine for all tests.
        """
        pass
    
    def testStateCompressionNode1(self, level=1):
        """ Test for StateCompressionNode.
        """
        testsig = np.array([[1.,1.],[2,2],[3,3],[4,4],[5,5],[6,6]])
        
        node = StateCompressionNode(2, 2, support_points=3)
        target = np.array([[2.,2.,4.,4.,6.,6.]])
        
        # simulate node
        testout = node(testsig)
        assert_array_almost_equal(testout, target)

    def testStateCompressionNode2(self, level=1):
        """ Test for StateCompressionNode, with linear interpolation
        """
        testsig = np.array([[1.,1.],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]])
        
        node = StateCompressionNode(2, 2, support_points=3)
        target = np.array([[2.333, 2.333, 4.667, 4.667, 7., 7.]])
        
        # simulate node
        testout = node(testsig)
        assert_array_almost_equal(testout, target, 2)
    
    def testVoteAverageNode(self, level=1):
        """ Test for VoteAverageNode.
        """
        testsig = np.array([[1., 2., 3., 4., 5., 6.],
                            [10, 11, 12, 13, 14, 15]])
        
        node = VoteAverageNode(6,3)
        target = np.array([[2.5, 3.5, 4.5],
                           [11.5, 12.5, 13.5]])
        
        # simulate node
        testout = node(testsig)
        assert_array_almost_equal(testout, target)
    
    def testSquareStatesNode(self, level=1):
        """ Test for SquareStatesNode.
        """
        testsig = np.array([[1., 2., 3., 4., 5., 6.],
                            [10, 11, 12, 13, 14, 15]]).T
        target = np.array([[1., 2., 3., 4., 5., 6.],
                           [10, 11, 12, 13, 14, 15],
                           [1.**2, 2.**2, 3.**2, 4.**2, 5.**2, 6.**2],
                           [10**2, 11**2, 12**2, 13**2, 14**2, 15**2]]).T
        
        node = SquareStatesNode(2)

        # simulate node
        testout = node(testsig)
        assert_array_almost_equal(testout, target)

#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    NumpyTest().run()
