import mdp
import unittest
import Engine

class MixinTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_washout_mixin_unsup(self):
        x = mdp.numx.zeros((100,1000))        
        Engine.utils.enable_washout(mdp.nodes.PCANode)
        pca = mdp.nodes.PCANode()
        pca.washout = 10
        
        assert(pca.is_trainable())
        y = pca(x)
                       
        assert(x.shape[0] == y.shape[0])        

    def test_washout_mixin_sup(self):
        x = mdp.numx.zeros((100,1000))        
        y = mdp.numx.zeros((100,1000))        
        Engine.utils.enable_washout(mdp.nodes.LinearRegressionNode, 10)
        lin = mdp.nodes.LinearRegressionNode()
        
        assert(lin.is_trainable())
        lin.train(x, y=y)
