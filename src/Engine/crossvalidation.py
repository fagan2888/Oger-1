'''
Created on Aug 25, 2009

@author: dvrstrae
'''
import mdp

def cross_val_random(inputs, n_folds):
    '''
    cross_val_random(inputs, n_folds) -> train_indices, test_indices
    
    Return indices to do random n_fold cross_validation. Two lists are returned, with n_folds elements each.
    - train_indices contains the indices of the dataset used for training
    - test_indices contains the indices of the dataset used for testing
    '''
    n_samples = len(inputs)
    # Create random permutation of number of samples
    randperm = mdp.numx.random.permutation(n_samples)
    train_indices, test_indices = [], []
    foldsize = mdp.numx.ceil(float(n_samples)/n_folds)
    for fold in range(n_folds):
        test_indices.append(randperm[fold*foldsize:foldsize*(fold+1)])
        train_indices.append(mdp.numx.setdiff1d(randperm, test_indices[-1]))
    return train_indices, test_indices
    

def cross_validate(inputs, outputs, flownode, error_measure, n_folds, cross_validate_function=cross_val_random):
    '''
    cross_validate(inputs, outputs, flownode, error_measure, n_folds, cross_validate_function=cross_val_random)
    
    Perform n_fold cross-validation on a flownode, return the validation error for each fold.
    - inputs and outputs are lists of arrays
    - flownode is an MDP flownode
    - error_measure is a function which should return a scalar
    - cross_validate_function is a function which determines the type of cross-validation
      Possible values are:
          - cross_val_random (default): split dataset in n_folds parts, for each fold train on n_folds-1 parts and test on the remainder
          - ...
    
    '''
    error = []
    train_indices, test_indices = cross_validate_function(inputs, n_folds)
    print "Performing cross-validation..."
    for fold in mdp.utils.progressinfo(range(n_folds), style='timer'):
        fnode = flownode.copy()
        [fnode.train(inputs[i], outputs[i]) for i in train_indices[fold]]
        fnode.stop_training()
        for i in test_indices[fold]:
            error.append(error_measure(fnode(inputs[i]), outputs[i]))
    return error





