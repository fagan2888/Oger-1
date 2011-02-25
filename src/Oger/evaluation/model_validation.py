import mdp
import Oger

def train_test_only(n_samples, training_fraction, random=True):
    '''
    train_test_only(n_samples, training_fraction, random) -> train_indices, test_indices
    
    Return indices to do simple training and testing. Only one fold is created, 
    using training_fraction of the dataset for training and the rest for testing.
    The samples are selected randomly by default but this can be disabled.
    Two lists are returned, with 1 element each.
        - train_indices contains the indices of the dataset used for training
        - test_indices contains the indices of the dataset used for testing
    '''
    if random:
        # Shuffle the samples randomly
        perm = mdp.numx.random.permutation(n_samples)
    else:
        perm = range(n_samples)
    # Take a fraction training_fraction for training
    train_indices = [perm[0:int(round(n_samples * training_fraction))]]
    # Use the rest for testing
    test_indices = mdp.numx.array([mdp.numx.setdiff1d(perm, train_indices[-1])])
    return train_indices, test_indices


def leave_one_out(n_samples):
    '''
    leave_one_out(n_samples) -> train_indices, test_indices
    
    Return indices to do leave-one-out cross-validation. Per fold, one example is used for testing and the rest for training.
    Two lists are returned, with n_samples elements each.
        - train_indices contains the indices of the dataset used for training
        - test_indices contains the indices of the dataset used for testing
    '''
    train_indices, test_indices = [], []
    all_samples = range(n_samples)
    # Loop over all sample indices, using each one for testing once
    for test_index in all_samples:
        test_indices.append(mdp.numx.array([test_index]))
        train_indices.append(mdp.numx.setdiff1d(all_samples, [test_index]))
    return train_indices, test_indices


def n_fold_random(n_samples, n_folds):
    '''
    n_fold_random(n_samples, n_folds) -> train_indices, test_indices
    
    Return indices to do random n_fold cross_validation. Two lists are returned, with n_folds elements each.
        - train_indices contains the indices of the dataset used for training
        - test_indices contains the indices of the dataset used for testing
    '''
    
    if n_folds < 2:
        raise RuntimeError('The number of folds should be > 1.')
    
    # Create random permutation of number of samples
    randperm = mdp.numx.random.permutation(n_samples)
    train_indices, test_indices = [], []
    foldsize = mdp.numx.floor(float(n_samples) / n_folds)
    
    if n_folds <= 1:
        raise Exception('Number of folds should be larger than one.')
    
    for fold in range(n_folds):
        # Select the sample indices used for testing
        test_indices.append(randperm[fold * foldsize:foldsize * (fold + 1)])
        # Select the rest for training
        train_indices.append(mdp.numx.array(mdp.numx.setdiff1d(randperm, test_indices[-1])))
    return train_indices, test_indices
    

def validate(data, flow, error_measure, cross_validate_function=n_fold_random, progress=True, *args, **kwargs):
    '''
    validate(data, flow, error_measure, cross_validate_function=n_fold_random, progress=True, *args, **kwargs) -> test_errors
    
    Perform  cross-validation on a flow, return the validation test_error for each fold. For every flow, the flow.train() method is called
    on the training data, and the flow.execute() function is called on the test data.
        - inputs and outputs are lists of arrays
        - flow is an mdp.Flow
        - error_measure is a function which should return a scalar
        - cross_validate_function is a function which determines the type of cross-validation
          Possible values are:
              - n_fold_random (default): split dataset in n_folds parts, for each fold train on n_folds-1 parts and test on the remainder
              - leave_one_out : do cross-validation with len(inputs) folds, using a single sample for testing in each fold and the rest for training
              - train_test_only : divide dataset into train- and testset, using training_fraction as the fraction of samples used for training
        - progress is a boolean to enable a progress bar (default True)
    '''
    test_error = []
    
    for f_copy, train_data, test_sample_list in validate_gen(data, flow, cross_validate_function, progress, *args, **kwargs):
        # Empty list to store test errors for current fold
        fold_error = []
        # test on all test samples
        for test_sample in test_sample_list:
            # If the last node is a feedback node: 
            if isinstance(f_copy[-1], Oger.nodes.FeedbackNode):
                for i in range(len(f_copy)):
                    # Disable state resetting for all reservoir nodes.
                    if isinstance(f_copy[i], Oger.nodes.ReservoirNode):
                        f_copy[i].reset_states = False       
                    if isinstance(f_copy[i], Oger.nodes.FeedbackNode):
                        f_copy[i].reset()
                               
                # Run flow on training data so we have an initial state to start from
                f_copy(train_data[0])     
                
                # TODO: the feedback node gets initiated with the last *estimated* timestep,
                # not the last training example. Fixing this will improve performance!
            
            fold_error.append(error_measure(f_copy(test_sample[-1][0][0]), test_sample[-1][0][-1]))
        test_error.append(mdp.numx.mean(fold_error))
        
    return test_error



def validate_gen(data, flow, cross_validate_function=n_fold_random, progress=True, *args, **kwargs):
    '''
    validate_gen(data, flow, cross_validate_function=n_fold_random, progress=True, *args, **kwargs) -> test_output
    
    This generator performs cross-validation on a flow. It splits the data into folds according to the supplied cross_validate_function, and then for each fold, trains the flow and yields the trained flow, the training data, and a list of test data samples.
    
    Use it like this:
    
    for flow, train_data, test_sample_list in validate_gen(...):
        ...
        
    See 'validate' for more information about the function signature.
    '''
    # Get the number of samples 
    n_samples = mdp.numx.amax(map(len, data))
    # Get the indices of the training and testing samples for each fold by calling the 
    # cross_validate_function hook
    
    train_samples, test_samples = cross_validate_function(n_samples, *args, **kwargs)
    
    if progress:
        print "Performing cross-validation using " + cross_validate_function.__name__
        iteration = mdp.utils.progressinfo(range(len(train_samples)), style='timer')
    else:
        iteration = range(len(train_samples))
        
    for fold in iteration:
        # Get the training data from the whole data set
        train_data = data_subset(data, train_samples[fold])
        # Empty list to store test errors for current fold
        fold_test = []
        
        # Copy the flow so we can re-train it for every fold
        # Only nodes that need training are copied.
        f_copy = mdp.Flow([])
        for node in flow:
            # TODO: check if this also works for e.g. LayerNodes with trainable
            # nodes inside
            if node.is_trainable():
                f_copy += node.copy()
            else:
                f_copy += node 
                
        # train on all training samples
        f_copy.train(train_data)
        
        test_sample_list = [data_subset(data, [k]) for k in test_samples[fold]]       
        yield (f_copy, train_data, test_sample_list) # collect everything needed to evaluate this fold and return it.


def data_subset(data, data_indices):
    '''
    data_subset(data, data_indices) -> data_subset
    
    Return a subset of the examples in data given by data_indices.
    Data_indices can be a slice, a list of scalars or a numpy array.
    '''
    n_nodes = len(data)
    subset = []
    #print data_indices
    #reprint type(data_indices)
    for node in range(n_nodes):
        if isinstance(data_indices, slice) or isinstance(data_indices, int):
            subset.append(data[node].__getitem__(data_indices))
        else:
            tmp_data = []
            if not data[node] == []:
                for data_index in data_indices:
                    tmp_data.append(data[node][data_index])
            subset.append(tmp_data)
    return subset
