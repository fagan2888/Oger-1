import mdp
import nose
import unittest
import Oger
import scipy as sp

class FunctionNodeTest(unittest.TestCase):
    def test_apply(self):
        f = Oger.nodes.FunctionNode(lambda x: x ** 2)
        x = sp.random.rand(5,5)
        assert(sp.all((f(x)== x**2)))

        def square(x):
            return x*x
        f = Oger.nodes.FunctionNode(square)
        x = sp.random.rand(5,5)
        assert(sp.all((f(x)== x**2)))

    def test_exception_passing(self):
        f = Oger.nodes.FunctionNode(sp.linalg.inv)
        x = sp.zeros((1,1))
        nose.tools.assert_raises(Exception, f, x)
