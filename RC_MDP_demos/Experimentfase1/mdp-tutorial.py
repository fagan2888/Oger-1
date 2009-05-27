#===============================================================================
# Some examples from the MDP tutorial at
# http://mdp-toolkit.sourceforge.net/tutorial.html
#===============================================================================

from numpy import *
import pylab
import mdp
import sys
import aureservoir as au
sys.path.append("../grhlib")
from reservoir_nodes import ReservoirNode

def pca_example():
    x = mdp.numx_rand.random((100, 25))  # 25 variables, 100 observations
    
    # estimate mean and covariance matrix
    pcanode1 = mdp.nodes.PCANode()
    pcanode1.train(x)
    
    # computes the eigenvectors
    pcanode1.stop_training()
    print "outputs: ", pcanode1.output_dim
    print "explained variance: ", pcanode1.explained_variance
    
    # get trained data
    avg = pcanode1.avg            # mean of the input data
    v = pcanode1.get_projmatrix() # projection matrix
    print "mean: ", avg
    print "projection matrix: ", v
    
    # execute the node (with .execute or just call the node):
    # The input data is projected on the principal components
    # learned in the training phase
#    x = mdp.numx_rand.random((100, 25))
    y_pca = pcanode1.execute(x)
    
    # invert the algorithm again
    print "PCA is invertible:", pcanode1.is_invertible()
    x_inv = pcanode1.inverse(y_pca)

    pylab.plot(x[0])
    pylab.plot(x_inv[0])
    pylab.plot(y_pca[1])
    pylab.show()
    

def supervised_training():
    fdanode = mdp.nodes.FDANode()
    x = {}
    
    # training with some labels
    for label in ['a', 'b', 'c']:
        x[label] = mdp.numx_rand.random((100, 25))
        fdanode.train(x[label], label)
    fdanode.stop_training()
    
    # possible second training phase
    for label in ['a', 'b', 'c']:
        fdanode.train(x[label], label)
    fdanode.stop_training()
    
    # The input data is projected to the directions learned by FDA:
    y_fda = fdanode.execute(x['a'])
    pylab.plot(y_fda[0])
    pylab.show()


class MeanFreeNode(mdp.Node):
    """ example custom node, derived from mdp.Node """
    
    def __init__(self, input_dim=None, dtype=None):
        super(MeanFreeNode, self).__init__(input_dim=input_dim,dtype=dtype)
        self.avg = None  # the mean
        self.tlen = 0    # nr of training points
    
    def _train(self, x):
        # Initialize the mean vector with the right
        # size and dtype if necessary:
        if self.avg is None:
            self.avg = mdp.numx.zeros(self.input_dim,dtype=self.dtype)
        
        self.avg += mdp.numx.sum(x, axis=0)
        self.tlen += x.shape[0]

    def _stop_training(self):
        self.avg /= self.tlen
        if self.output_dim is None:
            self.output_dim = self.input_dim
    
    def _execute(self, x):
        return x - self.avg
    
    def _inverse(self, y):
        return y + self.avg

def test_meanfree_node():
    node = MeanFreeNode()
    x = mdp.numx_rand.random((10,4))
    node.train(x)
    y = node(x)
    print 'Mean of y (should be zero): ', mdp.numx.mean(y, 0)


def logistic_map(x,r):
    return r*x*(1-x)

def logistic_map_sfa():
    """ extracts the slowly varying features of some functions """
    
    # slowly varying driving force: a combination of three sine waves
    p2 = mdp.numx.pi*2
    t = mdp.numx.linspace(0,1,10000,endpoint=0) # time axis 1s, samplerate 10KHz
    dforce = mdp.numx.sin(p2*5*t) + mdp.numx.sin(p2*11*t) + mdp.numx.sin(p2*13*t)
    
    # input timeseries: variables on columns and observations on rows
    series = mdp.numx.zeros((10000,1),'d')
    series[0] = 0.6
    for i in range(1,10000):
        series[i] = logistic_map(series[i-1],3.6+0.13*dforce[i])

#    pylab.plot( series, '.' )
#    pylab.show()
    
    # Flow to perform SFA in the space of polynomials of degree 3
    # EtaComputerNode: measures slowness
    # TimeFramesNode: embeds the 1-dimensional time series in a 10 dimensional
    #                 space using a sliding temporal window of size 10
#    flow = mdp.Flow([mdp.nodes.TimeFramesNode(10), \
#                     mdp.nodes.PolynomialExpansionNode(3), \
#                     mdp.nodes.SFANode(output_dim=3)] )
    
    # reservoir prototype
    prot = au.DoubleESN()
    prot.setInitParam( au.CONNECTIVITY, 0.1 )
    prot.setInitParam( au.ALPHA, 0.9 )
    prot.setSize(100)
    
    # Reservoir Node with SFA
    flow = mdp.Flow([ReservoirNode(1, 100, prototype=prot), \
                        #mdp.nodes.PCANode(output_dim=30,svd=True), \
#                        mdp.nodes.PolynomialExpansionNode(2), \
                        mdp.nodes.SFANode(output_dim=3)])
    
    flow.train(series)
    
    # execute flow to get the slow features
    slow = flow(series)
    
    # rescale driving force for comparison
    resc_dforce = (dforce - mdp.numx.mean(dforce,0))/mdp.numx.std(dforce,0)
    
#    print 'Eta value (time series): ', flow[0].get_eta(t=10000)
#    print 'Eta value (slow feature): ', flow[4].get_eta(t=9990)
    
    pylab.plot( slow[:,0] )
#    pylab.plot( slow[:,1] )
#    pylab.plot( slow[:,2] )
    pylab.plot( resc_dforce )
    pylab.show()
    


    


#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
#    pca_example()
#    supervised_training()
#    test_meanfree_node()
    logistic_map_sfa()
