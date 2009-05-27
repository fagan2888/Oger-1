#===============================================================================
# Test scripts of self-made nodes for the MDP package
#===============================================================================

import sys
from numpy.testing import *
import numpy as np
import mdp
import svm
sys.path.append("../")
from svm_nodes import BinarySVMNode, BinaryLinearSVMNode


class test_svm(NumpyTestCase):
    """ Tests svm_nodes.
    """

    def setUp(self):
        """ Setup routine for all tests.
        """
        # data for single class experiments
        self.sc_labels = np.array([0, 0, 1, 1]).reshape(-1,1)
        self.sc_samples = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        
        # data for multiclass experiments
        self.mc_labels = np.array([[-1, 1, -1,  1],
                                   [ 1, 1, -1, -1],
                                   [ 1, 1,  1,  1],
                                   [-1, 1,  1, -1]])
        self.mc_samples = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
            
    def testSingleClass(self, level=1):
        """ Single class classification test with BinarySVMNode.
        """
        params = svm.svm_parameter(kernel_type = svm.RBF, C = 10)
        node = BinarySVMNode(2,1,params)
        node.train(self.sc_samples, self.sc_labels)
        node.stop_training()
        
        testresult = node(self.sc_samples)
        
        # rescale from SVM output [-1,1] to [0,1]
        testresult = (testresult+1) / 2.
        
        # test if labels are the same as the test output
        assert_array_almost_equal(self.sc_labels, testresult, 2)

    def testMultiClass(self, level=1):
        """ Multiclass classification test with BinarySVM Node.
        """
        params = svm.svm_parameter(kernel_type = svm.RBF, C = 10)
        node = BinarySVMNode(2,4,params)
        node.train(self.mc_samples, self.mc_labels)
        node.stop_training()
        
        testresult = node(self.mc_samples)
        
        # test if labels are the same as the test output
        assert_array_almost_equal(self.mc_labels, testresult, 2)

    def testSingleClassLinear(self, level=1):
        """ Single class classification test with BinaryLinearSVMNode.
        """
        node = BinaryLinearSVMNode(2,1,C=2)
        node.train(self.sc_samples, self.sc_labels)
        node.stop_training()
        
        testresult = node(self.sc_samples)

        target = np.array([[-0.68643796],
                           [-0.75505737],
                           [ 0.8447326 ],
                           [ 0.77611319]])
        # test if labels are the same as the test output
        assert_array_almost_equal(target, testresult, 2)

    def testMultiClassLinear(self, level=1):
        """ Multiclass classification test with BinaryLinearSVM Node.
        """
        node = BinaryLinearSVMNode(2,4,C=2)
        node.train(self.mc_samples, self.mc_labels)
        node.stop_training()
        
        testresult = node(self.mc_samples)
        
        target = np.array([[-0.0140, 1., -0.6870,  0.6869],
                           [-0.0104, 1., -0.7572, -0.8429],
                           [-0.0158, 1.,  0.8431,  0.7569],
                           [-0.0122, 1.,  0.7729, -0.7729]])
        # test if labels are the same as the test output
        assert_array_almost_equal(target, testresult, 1)

#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    NumpyTest().run()
