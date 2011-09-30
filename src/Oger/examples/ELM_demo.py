# This is a demo for an OP-ELM that tries to model a sinc function

import mdp
import Oger
import numpy as np
import pylab

X_orr = np.atleast_2d(np.arange(-5, 5, 0.025)).T
Y_orr = np.sinc(X_orr)
P = np.random.permutation(len(X_orr))
X, Y = X_orr[P,:], Y_orr[P,:] + (np.random.rand(len(X_orr),1)-.5)/4

n_folds = 20
l = np.floor(len(X)/n_folds)
x,y=[],[]
for i in range(n_folds-1):
    x.append(X[i*l:(i+1)*l,:])
    y.append(Y[i*l:(i+1)*l,:])

#test data
x_t = X[(n_folds-1)*l:,:]
y_t = Y_orr[P[(n_folds-1)*l:],:]


elm = Oger.nodes.ELMNode()
readout = Oger.nodes.OPRidgeRegressionNode(ridge_param=0)
flow = mdp.Flow([elm, readout],verbose=True)
flow.train([x, zip(x,y)])
yh_test = flow(x_t)
print '\nTest NRMSE =', Oger.utils.nrmse(yh_test, y_t), '\n'

pylab.scatter(X,Y, c='k', marker='+')
pylab.plot(X_orr, Y_orr, linewidth=2)
pylab.scatter(x_t,yh_test,c='r',linewidths=2)
pylab.show()