import mdp, scipy
import Oger
from numpy import *
from scipy import randn, tile, concatenate, ones, zeros, mean

import matplotlib.pyplot
from matplotlib.pyplot import figure, plot, hold, show, gca

def generate_cluster (c, N):
    return randn(N, 2)/10. + tile(c, (N, 1));

# create 2 sets of points for 2 clusters (one training, one testing)
a_train = generate_cluster([0.1,0.2], 100)
b_train = generate_cluster([0.3,0.5], 1250) #if this gets bigger cg will fail...

a_test = generate_cluster([0.1,0.2], 20)
b_test = generate_cluster([0.3,0.5], 20)

xtrain = concatenate((a_train, b_train))
xtest = concatenate((a_test, b_test))
ytrain = concatenate((ones(a_train.shape[0]), zeros(b_train.shape[0])));
ytest = concatenate((ones(a_test.shape[0]), zeros(b_test.shape[0])));

## Test the IRLS node
node = Oger.nodes.IRLSLogisticRegressionNode()
flow = mdp.Flow([node])

# train the logreg node
flow.train([zip([xtrain], [ytrain])])

# test the logreg node
y_irls = flow.execute(xtest).flatten()

# calculate the error
y_irls[y_irls>=0.5] = 1
y_irls[y_irls<0.5] = 0
print "error rate irls: ", sum(abs(ytest - y_irls)) / ytest.shape[0]

# Test CG node
# change the gtrainer in the following line to test other algorithms such as BFGS, ... 
cgnode = Oger.gradient.LogisticRegressionNode(input_dim=2, output_dim=1, gtrainer=Oger.gradient.CGTrainer())
flow = mdp.Flow([cgnode])
flow.train([zip([xtrain], [array(matrix(ytrain).T)])])
y_cg = flow.execute(xtest)
y_cg[y_cg>=0.5] = 1
y_cg[y_cg<0.5] = 0

print "error rate cg: ", sum(abs(ytest - y_cg.flatten())) / ytest.shape[0]

# plot the results
figure()
plot(
     xtrain[ytrain==1, 0], xtrain[ytrain==1, 1], 'g*',
     xtrain[ytrain==0, 0], xtrain[ytrain==0, 1], 'rd',
     xtest[ytest==1, 0], xtest[ytest==1, 1], 'go',
     xtest[ytest==0, 0], xtest[ytest==0, 1], 'rs',
     xtest[y_irls!=ytest, 0], xtest[y_irls!=ytest, 1], '+',
     xtest[y_cg.flatten() != ytest, 0], xtest[y_cg.flatten() != ytest, 1], 'x')

# plot the boundaries
r = gca().get_xlim()

y1 = (node.w[0] * r[0] + node.b) / -node.w[1]
y2 = (node.w[0] * r[1] + node.b) / -node.w[1]
plot(r,[y1, y2], 'b')

y1 = (cgnode.perceptron.w[0] * r[0] + cgnode.perceptron.b) / -cgnode.perceptron.w[1]
y2 = (cgnode.perceptron.w[0] * r[1] + cgnode.perceptron.b) / -cgnode.perceptron.w[1]
plot(r,[y1, y2], 'r')

show()
