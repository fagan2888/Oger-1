#===============================================================================
# playing aroud with libsvm and PyML ...
#===============================================================================

import svm
import liblinear as ll
from numpy import *

# a two-class problem
#labels = array([0., 1., 1., 2.])
labels = array([-1, 1, 1, -1])
samples = array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])

# set the parameters of the SVM
param = svm.svm_parameter(kernel_type = svm.LINEAR, C = 10)
param.kernel_type = svm.RBF

# svm_problem is used to hold the training data for the problem
prob = svm.svm_problem(labels, samples)

# now construct the model
model = svm.svm_model(prob, param)
print "Number of classes:", model.get_nr_class()

# predict one new sample with the model:
#testdata = array([1., 0.])
testdata = array([[1., 0.], [1., 1.], [0., 0.], [0., 1.]])
for data in testdata:
    print "One Prediction: ",model.predict( data )
    print "Desicion Values of the Prediction: ",model.predict_values( data )#[(1,-1)]
#    print "Probability of the Prediction: ",model.predict_probability( data )



print "---------LIBLINEAR----------------"

class1 = [ll.vector2sparse(samples[0]), ll.vector2sparse(samples[1])]
class2 = [ll.vector2sparse(samples[2]), ll.vector2sparse(samples[3])]
print "class1:",class1
print "class2:",class2

data = [class1, class2]
clfr = ll.LinearSVM.train(data, C=2., mach=ll.L2LOSS_SVM )

# predict one new sample with the model:
for dp in testdata:
    print "Prediction of",dp,":",clfr.predict( ll.vector2sparse(dp) )
    print "Pr.-Values of",dp,":",clfr.predict_values( ll.vector2sparse(dp) )
#p1 = clfr.predict_probabilites(good_vec1)

