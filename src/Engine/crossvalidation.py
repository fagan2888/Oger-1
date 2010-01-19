'''
Created on Aug 25, 2009

@author: dvrstrae
'''
#import mdp
import scipy as sp

def cross_val_random(inputs, n_folds):
    n_samples = len(inputs)
    # Create random permutation of number of samples 
    randperm = sp.random.permutation(n_samples)
    print randperm
    train_indices, test_indices = [], []
    foldsize = round(n_samples/n_folds)
    print foldsize
    for fold in range(n_folds):
        print foldsize
        print fold
        len(randperm)
        print randperm[fold*foldsize:foldsize*(fold+1)]
        train_indices.append(randperm[fold*foldsize:foldsize*(fold+1)])
        test_indices.append(sp.setdiff1d(randperm, train_indices))
        print train_indices(fold)
        print test_indices(fold)

def cross_validate(inputs, outputs, flownode, error_measure, n_folds, cross_validate_function=cross_val_random):
    train_indices, test_indices = cross_validate_function(inputs, n_folds);
    print train_indices
    print test_indices
#    for fold in range(n_folds):
#        print 'Fold nr. ' + fold
#        for xt,yt in mdp.utils.progressinfo(zip(inputs[0:n_train_samples-1], outputs[0:n_train_samples-1])):
#            flownode.train(xt,yt)
#        flownode.stop_training()





