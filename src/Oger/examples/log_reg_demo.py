## A demo to demonstrate the different implemented logistic regression methods
# convergence time might be very long for some algorithms...

import mdp
import Oger
import scipy as sp
import pylab

def generate_cluster (c, N):
    return sp.random.randn(N, 2) / 10. + sp.tile(c, (N, 1));

# create 2 sets of points for 2 clusters (one training, one testing)
a_train = generate_cluster([.1, .2], 10)
b_train = generate_cluster([.3, .5], 100)

a_test = generate_cluster([.1, .2], 20)
b_test = generate_cluster([.3, .5], 20)

xtrain = sp.concatenate((a_train, b_train))
xtest = sp.concatenate((a_test, b_test))
ytrain = sp.concatenate((sp.ones(a_train.shape[0]), sp.zeros(b_train.shape[0]))).reshape((-1, 1))
ytest = sp.concatenate((sp.ones(a_test.shape[0]), sp.zeros(b_test.shape[0]))).reshape((-1, 1))

## Test the IRLS node
node = Oger.nodes.RidgeRegressionNode()
flow = mdp.Flow([node])

# train the logreg node
flow.train([zip([xtrain], [ytrain])])

# test the logreg node
y_lin = flow.execute(xtest)

# calculate the error
y_lin[y_lin >= 0.5] = 1
y_lin[y_lin < 0.5] = 0
print "error rate lin regr: ", sum(abs(ytest - y_lin)) / ytest.shape[0]


## Test the IRLS node
irls_node = Oger.nodes.IRLSLogisticRegressionNode()
flow = mdp.Flow([irls_node])

# train the logreg node
flow.train([zip([xtrain], [ytrain])])

# test the logreg node
y_irls = flow.execute(xtest)

# calculate the error
y_irls[y_irls >= 0.5] = 1
y_irls[y_irls < 0.5] = 0
print "error rate irls: ", sum(abs(ytest - y_irls)) / ytest.shape[0]


## Test the GD node
gdnode = Oger.nodes.LogisticRegressionNode(Oger.gradient.GradientDescentTrainer(), 'epochs', 200)
flow = mdp.Flow([gdnode])

# train the logreg node
flow.train([zip([xtrain], [ytrain])])

# test the logreg node
y_gd = flow.execute(xtest)

# calculate the error
y_gd[y_gd >= 0.5] = 1
y_gd[y_gd < 0.5] = 0
print "error rate gd: ", sum(abs(ytest - y_gd)) / ytest.shape[0]


# Test the logistic regression node with CG training and linear regression initialization (default)
# sometimes it converges really slow... IRLS is often faster...
cg_node = Oger.nodes.LogisticRegressionNode(Oger.gradient.CGTrainer())
flow = mdp.Flow([cg_node])

# train the logreg node
flow.train([zip([xtrain], [ytrain])])

# test the logreg node
y_cg = flow.execute(xtest)

# calculate the error
y_cg[y_cg >= 0.5] = 1
y_cg[y_cg < 0.5] = 0
print "error rate cg: ", sum(abs(ytest - y_cg)) / ytest.shape[0]


# Test the logistic regression node with BFGS training and linear regression initialization (default)
# sometimes it converges really slow...IRLS is often faster...
bfgs_node = Oger.nodes.LogisticRegressionNode(Oger.gradient.BFGSTrainer())
flow = mdp.Flow([bfgs_node])

# train the logreg node
flow.train([zip([xtrain], [ytrain])])

# test the logreg node
y_bfgs = flow.execute(xtest)

# calculate the error
y_bfgs[y_bfgs >= 0.5] = 1
y_bfgs[y_bfgs < 0.5] = 0
print "error rate bfgs: ", sum(abs(ytest - y_bfgs)) / ytest.shape[0]



# plot the results
pylab.figure()
pylab.plot(
     xtrain[ytrain.flatten() == 1, 0], xtrain[ytrain.flatten() == 1, 1], 'g*',
     xtrain[ytrain.flatten() == 0, 0], xtrain[ytrain.flatten() == 0, 1], 'rd',
     xtest[ytest.flatten() == 1, 0], xtest[ytest.flatten() == 1, 1], 'go',
     xtest[ytest.flatten() == 0, 0], xtest[ytest.flatten() == 0, 1], 'rs',
     xtest[y_lin.flatten() != ytest.flatten(), 0], xtest[y_lin.flatten() != ytest.flatten(), 1], 'k+',
     xtest[y_cg.flatten() != ytest.flatten(), 0], xtest[y_cg.flatten() != ytest.flatten(), 1], 'kx')

# plot the boundaries
r = pylab.gca().get_xlim()

y1 = (node.beta[1] * r[0] + node.beta[0] - .5) / -node.beta[2]
y2 = (node.beta[1] * r[1] + node.beta[0] - .5) / -node.beta[2]
pylab.plot(r, [y1, y2], 'r')

y1 = (irls_node.w[0] * r[0] + irls_node.b) / -irls_node.w[1]
y2 = (irls_node.w[0] * r[1] + irls_node.b) / -irls_node.w[1]
pylab.plot(r, [y1, y2], 'b')

y1 = (gdnode.w[0] * r[0] + gdnode.b) / -gdnode.w[1]
y2 = (gdnode.w[0] * r[1] + gdnode.b) / -gdnode.w[1]
pylab.plot(r, [y1, y2], 'g')

y1 = (cg_node.w[0] * r[0] + cg_node.b) / -cg_node.w[1]
y2 = (cg_node.w[0] * r[1] + cg_node.b) / -cg_node.w[1]
pylab.plot(r, [y1, y2], 'k')

y1 = (bfgs_node.w[0] * r[0] + bfgs_node.b) / -bfgs_node.w[1]
y2 = (bfgs_node.w[0] * r[1] + bfgs_node.b) / -bfgs_node.w[1]
pylab.plot(r, [y1, y2], 'y')

pylab.show()
