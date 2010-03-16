'''
Created on Dec 7, 2009

@author: dvrstrae
'''

import sys
import mdp
import nose
import unittest
import reservoir_nodes
import utility_functions
import scipy as sp

class ReservoirNodeTest(unittest.TestCase):
    def setUp(self):
        self.default_size = 10
        self.input_length = 10

    def test_spectral_radius(self):
        ''' Test if spectral radius is expected value '''

        # Default value
        r = reservoir_nodes.ReservoirNode (output_dim = self.default_size)
        nose.tools.assert_almost_equals(utility_functions.get_spectral_radius(r.w), sp.amax(sp.absolute(sp.linalg.eigvals(r.w)))) 

        # Custom value
        rho = 0.1
        r = reservoir_nodes.ReservoirNode (output_dim = self.default_size, spectral_radius=rho)
        nose.tools.assert_almost_equals(utility_functions.get_spectral_radius(r.w), rho) 

    def test_zero_input(self):
        ''' Test if zero input returns zero output without bias
        '''
        r = reservoir_nodes.ReservoirNode (output_dim = self.default_size)
        assert sp.all(r(sp.zeros((self.input_length,1))) == sp.zeros((self.input_length,self.default_size)))

    def test_input_mapping(self):
        ''' Test if input mapping is correct for zero internal weight matrix 
        '''
        r = reservoir_nodes.ReservoirNode (output_dim = self.default_size, spectral_radius=0)
        assert sp.all(r(sp.ones((self.input_length,1))) == sp.tanh(sp.dot(sp.ones((self.input_length,1)), r.w_in.T)))
        
    def test_bias(self):
        ''' Test if turning on bias gives expected results 
        '''
        r = reservoir_nodes.ReservoirNode (output_dim = self.default_size, spectral_radius=0, bias_scaling = 1)
        assert sp.all(r(sp.zeros((self.input_length,1))) == sp.tile(sp.tanh(r.w_bias), (self.input_length, 1)))


