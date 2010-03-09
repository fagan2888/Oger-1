'''
Created on Aug 25, 2009

@author: dvrstrae
'''
import mdp

def train_test_only(inputs, training_fraction):
    '''
    train_test_only(inputs, training_fraction) -> train_indices, test_indices
    
    Return indices to do simple training and testing. Only one fold is created, using training_fraction of the dataset for training and the rest for testing.
    The samples are selected randomly.
    Two lists are returned, with 1 element each.
    - train_indices contains the indices of the dataset used for training
    - test_indices contains the indices of the dataset used for testing
    '''
    n_samples = len(inputs)
    # Shuffle the samples
    randperm = mdp.numx.random.permutation(n_samples)
    # Take a fraction training_fraction for training
    train_indices = [randperm[0:int(round(n_samples*training_fraction))]]
    # Use the rest for testing
    test_indices = [mdp.numx.setdiff1d(randperm, train_indices[-1])]
    return train_indices, test_indices


def leave_one_out(inputs):
    '''
    leave_one_out(inputs) -> train_indices, test_indices
    
    Return indices to do leave-one-out cross-validation. Per fold, one example is used for testing and the rest for training.
    Two lists are returned, with len(inputs) elements each.
    - train_indices contains the indices of the dataset used for training
    - test_indices contains the indices of the dataset used for testing
    '''
    n_samples = len(inputs)
    train_indices, test_indices = [], []
    all_samples = range(n_samples)
    # Loop over all sample indices, using each one for testing once
    for test_index in all_samples:
        test_indices.append([test_index])
        train_indices.append(mdp.numx.setdiff1d(all_samples, [test_index]))
    return train_indices, test_indices


def n_fold_random(inputs, n_folds):
    '''
    n_fold_random(inputs, n_folds) -> train_indices, test_indices
    
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
        # Select the sample indices used for testing
        test_indices.append(randperm[fold*foldsize:foldsize*(fold+1)])
        # Select the rest for training
        train_indices.append(mdp.numx.setdiff1d(randperm, test_indices[-1]))
    return train_indices, test_indices
    

def cross_validate(inputs, outputs, flownode, error_measure, cross_validate_function=n_fold_random, *args, **kwargs):
    '''
    cross_validate(inputs, outputs, flownode, error_measure, cross_validate_function=n_fold_random *args, **kwargs)
    
    Perform  cross-validation on a flownode, return the validation test_error for each fold.
    - inputs and outputs are lists of arrays
    - flownode is an MDP flownode
    - error_measure is a function which should return a scalar
    - cross_validate_function is a function which determines the type of cross-validation
      Possible values are:
          - n_fold_random (default): split dataset in n_folds parts, for each fold train on n_folds-1 parts and test on the remainder
          - leave_one_out : do cross-validation with len(inputs) folds, using a single sample for testing in each fold and the rest for training
          - train_test_only : divide dataset into train- and testset, using training_fraction as the fraction of samples used for training
    
    '''
    test_error = []
    train_samples, test_samples = cross_validate_function(inputs, *args, **kwargs)
    print "Performing cross-validation using " + cross_validate_function.__name__
    for fold in mdp.utils.progressinfo(range(len(train_samples)), style='timer'):
        # Empty list to store test errors for current fold
        fold_error = []
        # Copy the flownode so we can re-train it for every fold
        fnode = flownode.copy()
        # train on all training samples
        for i in train_samples[fold]:
            fnode.train(inputs[i], outputs[i])
        fnode.stop_training()
        # test on all test samples
        for i in test_samples[fold]:
            fold_error.append(error_measure(fnode(inputs[i]), outputs[i]))
        test_error.append(mdp.numx.mean(fold_error))
    return test_error





