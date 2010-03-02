'''
Created on Dec 7, 2009

@author: dvrstrae
'''

import sys
import mdp
import nose
import Engine.reservoir_nodes

def reservoir_test():
    r = Engine.reservoir_nodes.ReservoirNode (output_dim = 10)
    #assert_almost_equals(E.utility_functions.get_spectral_radius(r.W), sp.max(sp.abs(sp.linalg.eigvals(r.W)))) 
