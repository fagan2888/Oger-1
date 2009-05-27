#===============================================================================
# Constructs data for a Classification Experiment
#===============================================================================

import shelve
import numpy as np


def add_length_feature(data):
    """ adds length of the file as a feature.
    """
    for n in range(len(data)):
        l = len(data[n])
        data[n] = np.c_[ data[n], l*np.ones((l,1)) ]

def add_bias_feature(data,bias):
    """ adds constant bias as a feature.
    """
    for n in range(len(data)):
        l = len(data[n])
        data[n] = np.c_[ data[n], bias*np.ones((l,1)) ]

def calc_minmax(data):
    """ Calculates the minimum and maximum of each channel.
    """
    channels = data[0].shape[1]
    
    # find minimum of each channel
    mins = np.ones(channels) * 10000
    maxs = np.ones(channels) * -10000
    
    for ex in data:
        for n in range(channels):
            min = ex[:,n].min()
            if min < mins[n]:
                mins[n] = min
            max = ex[:,n].max()
            if max > maxs[n]:
                maxs[n] = max
    
    return mins,maxs

def scale_data(data,mins,maxs,targ_mins,targ_maxs,scale_prop):
    """ Scales the data.

    mins         --   current minimums of each channel
    maxs         --   current maximums of each channel
    targ_mins    --   array with target minimums for each channel
    targ_maxs    --   array with target maximums for each channel
    scale_prop   --   scaling properties
    """
    channels = data[0].shape[1]
    
    # calculate scale and shift
    if scale_prop=="global":
        # scales channel to global min/max
        glob_max = maxs.max()
        glob_min = mins.min() 
        targ_max = targ_maxs.max()
        targ_min = targ_mins.min()
        
        glob_scale = (targ_max-targ_min) / (glob_max-glob_min)
        glob_shift = (-1)*glob_min * glob_scale + targ_min
        
        scale = np.ones(channels) * glob_scale
        shift = np.ones(channels) * glob_shift
        
    elif scale_prop=="local":
        # shift each channel to its individual min/max
        scale = (targ_maxs-targ_mins) / (maxs-mins)
        shift = (-1)*mins * scale + targ_mins

        # if infinity (same max as min, constant input)
        # we set the input to zero
        ind1 = np.nonzero(np.isinf(scale)==True)
        ind2 = np.nonzero(np.isinf(shift)==True)
        scale[ind1] = 0.
        shift[ind2] = 0
    
    else:
        # no scaling
        scale = np.ones(channels)
        shift = np.zeros(channels)
        return data,scale,shift

    # scale and shift data
    for ex in data:
        for n in range(channels):
            ex[:,n] = ex[:,n]*scale[n] + shift[n]
    
    return data,scale,shift

def divide_data(data,n=3,fold=0):
    """ data must be a list.
    
    n     --  nr of folds
    fold  --  current fold
    """
    leng = len(data)
    ind = range(leng)
    
    testind = ind[fold::n]
    trainind = ind
    for aa in testind:
        trainind.remove(aa)

    trainset = []
    testset = []
    for i in trainind:
        trainset.append( data[i] )
    for i in testind:
        testset.append( data[i] )

    return trainset, testset

def mk_target(data,targ):
    target = []
    for ex in data:
        leng = ex.shape[0]
        t = np.tile(targ, (leng,1))
        target.append(t)
    return target

def data_preprocessing(features,bias,scale_min,scale_max,add_length,scale_prop):
    """ Preprocesses the data.
    
    features   --   list with features.
    """
    # add length of the files as a feature
    if add_length:
        for ft in features:
            add_length_feature(ft)
    
    # SCALE DATA:
    # calc mins,maxs over all features
    all = []
    for ft in features:
        all += ft
    mins,maxs = calc_minmax(all)
    del all
    
    # new target mins,maxs
    channels = features[0][0].shape[1]
    targ_mins = np.ones(channels) * scale_min
    targ_maxs = np.ones(channels) * scale_max
    
    # finally scale and shift data
    for i in range(len(features)):
        features[i],sc,sh =  scale_data(features[i],mins,maxs,
                                        targ_mins,targ_maxs,scale_prop)
    
    # add bias input
    if bias > 1e-12:
        for i in range(len(features)):
            add_bias_feature(features[i],bias)


#===============================================================================
# high level interface
#===============================================================================

def binary_classification_1fold(class1_features,class2_features,
                                n=3,fold=0,label=0.9,bias=1.,
                                scale_min=0., scale_max=1.,
                                add_length=1, scale_prop="local",
                                feature_key="features"):
    """Generates data for a binary classification experiment with one fold.
    
    class1_features    --    file with features for the first class
    class2_features    --    file with features for the second class
    n     --   how to partition the training/test data (nr of folds)
    fold  --  current fold
    label --   value for the target output (class1 is +label,
               class2 is -label)
    bias  --   value of the bias input
    scale_min    --   minimum of each channel after scaling
    scale_max    --   maximum of each channel after scaling
    add_length   --   add length to feature vector
    scale_prop    --   different scaling methods:
                       "local"   --    scales each channel to its min/max
                       "global"  --    scales each channel to the global min/max
                       "noscale" --    no scaling at all
    feature_key  --   key for shelve data, to get the features
    """
    features = []
    data = shelve.open(class1_features)
    features.append( data[feature_key] )
    data.close()
    
    data = shelve.open(class2_features)
    features.append( data[feature_key] )
    data.close()
    
#    import pylab
#    pylab.plot(features[0][0])
    
    # make data preprocessing
    data_preprocessing(features,bias,scale_min,scale_max,add_length,scale_prop)
    
#    pylab.figure()
#    pylab.plot(features[0][0])
#    pylab.show()
#    exit(0)
    
    # make train and test sets
    set1 = divide_data(features[0],n,fold)
    set2 = divide_data(features[1],n,fold)
    
    trainin = set1[0] + set2[0]
    trainout = mk_target(set1[0], label*np.ones(1)) + mk_target(set2[0],
                                                          -1*label*np.ones(1))
    
    testin = set1[1] + set2[1]
    testout = mk_target(set1[1], label*np.ones(1)) + mk_target(set2[1],
                                                         -1*label*np.ones(1))
    
    return trainin, trainout, testin, testout

def multiclass_classification_1fold(feature_files,n=3,fold=0,label=0.9,bias=1.,
                                    scale_min=0., scale_max=1., add_length=1.,
                                    scale_prop="local", feature_key="features",
                                    label_key="label"):
    """ Performs one fold of a multiclass experiment.
    
    feature_files --   a list with all the feature files
    n             --   how to partition the training/test data (nr of folds)
    fold          --   current fold
    label         --   value for the target output (class1 is +label,
                       class2 is -label)
    bias          --   value of the bias input
    scale_min     --   minimum of each channel after scaling
    scale_max     --   maximum of each channel after scaling
    add_length    --   add length to feature vector
    scale_prop    --   different scaling methods:
                       "local"   --    scales each channel to its min/max
                       "global"  --    scales each channel to the global min/max
                       "noscale" --    no scaling at all
    feature_key   --   key for shelve data, to get the features
    label_key     --   key for shelve data, to get label of current file
    """
    # read all features
    features = []
    labels = []
    for file in feature_files:
        data = shelve.open(file)
        features.append(data[feature_key])
        labels.append(data[label_key])
        data.close()
    
    # make data preprocessing
    data_preprocessing(features,bias,scale_min,scale_max,add_length,scale_prop)

    # make train and test sets
    for i in range(len(features)):
        features[i] = divide_data(features[i],n,fold)
    
    # make train and test in
    trainin = []
    testin = []
    trainout = []
    testout = []
#    trainlabels = []
#    testlabels = []
    for i in range(len(features)):
        # train and testin sets
        trainin += features[i][0]
        testin += features[i][1]
        
        # make train and test target signal
        targ = np.ones(len(labels)) * (-1) * label
        targ[i] = label
        trainout += mk_target(features[i][0], targ)
        testout += mk_target(features[i][1], targ)

    return trainin, trainout, testin, testout

def multiclass_anytime(data_file,train_range,test_range,
                       label=0.9,bias=1.,
                       scale_min=0., scale_max=1.,scale_prop="local",
                       feature_key="features",target_key="targets"):
    """ Multiclass dataset for anytime classification (e.g. for melody extraction)
    for one single data file.
    
    data_file     --   file with the features and target data
    train_range   --   list with start and stop framenr to build the training set
    test_range    --   list with start and stop framenr to build the testing set
    label         --   value for the target output (class1 is +label,
                       class2 is -label)
    bias          --   value of the bias input
    scale_min     --   minimum of each channel after scaling
    scale_max     --   maximum of each channel after scaling
    scale_prop    --   different scaling methods:
                       "local"   --    scales each channel to its min/max
                       "global"  --    scales each channel to the global min/max
                       "noscale" --    no scaling at all
    feature_key   --   key for shelve data, to get the features
    target_key    --   key for shelve data, to get the targets
    """
    # read features and targets
    features = []
    targets = []
    data = shelve.open(data_file)
    feature = []
    feature.append(data[feature_key])
    features.append(feature) # because of the double list needed by data_preproc
    targets.append(data[target_key])
    data.close()
    
    # make data preprocessing
    data_preprocessing(features,bias,scale_min,scale_max,0,scale_prop)
    
    # scale targets
    mins,maxs = calc_minmax(targets)
    channels = len(mins)
    scale_data(targets,mins,maxs,(-1)*label*np.ones(channels),
               label*np.ones(channels),"local")
    
    # make train and test data
    trainin = []
    testin = []
    trainout = []
    testout = []
    trainin.append(features[0][0][train_range[0]:train_range[1]])
    trainout.append(targets[0][train_range[0]:train_range[1]])
    testin.append(features[0][0][test_range[0]:test_range[1]])
    testout.append(targets[0][test_range[0]:test_range[1]])

    return trainin, trainout, testin, testout

def multiclass_anytime2(data_files,train_range,test_range,
                       label=0.9,bias=1.,
                       scale_min=0., scale_max=1.,scale_prop="local",
                       feature_key="features",target_key="target_midi"):
    """ Like multiclass_anytime, but works with multiple files.
    
    data_files    --   files with the features and target data
    train_range   --   list with start and stop framenr to build the training set, and this
                       for each element in the data_files list
    test_range    --   list with start and stop framenr to build the testing set, and this
                       for each element in the data_files list
    label         --   value for the target output (class1 is +label,
                       class2 is -label)
    bias          --   value of the bias input
    scale_min     --   minimum of each channel after scaling
    scale_max     --   maximum of each channel after scaling
    scale_prop    --   different scaling methods:
                       "local"   --    scales each channel to its min/max
                       "global"  --    scales each channel to the global min/max
                       "noscale" --    no scaling at all
    feature_key   --   key for shelve data, to get the features
    target_key    --   key for shelve data, to get the targets
    """
    # read features and targets
    features = []
    targets = []
    feature = []
    target = []
    for file in data_files:
        data = shelve.open(file)
        print file,"feature shape:", data[feature_key].shape
        feature.append(data[feature_key])
        target.append(data[target_key])
        data.close()
    features.append(feature)
    targets.append(target)
    
    # make data preprocessing
    data_preprocessing(features,bias,scale_min,scale_max,0,scale_prop)

    # make targets
    
    # check how many pitch classes we have
    all_keys = []
    for el in targets[0]:
        all_keys += el.tolist()
    classes = list(set(all_keys))
    classes.sort()
    print "classes:", classes
    print "nr classes:",len(classes)

    # make (binary) target data
    cl_targets = []
    for piece in targets[0]:
        target = np.ones((len(piece), len(classes))) * (-1)*label
        for n in range(len(piece)):
            ind = classes.index( piece[n] )
            target[n,ind] = label
        cl_targets.append(target)
    
    # make train and test data
    trainin = []
    testin = []
    trainout = []
    testout = []
    nr_ex = len(train_range)
    for n in range(nr_ex):
        trainin.append( features[0][n][ train_range[n][0]:train_range[n][1] ] )
        trainout.append( cl_targets[n][ train_range[n][0]:train_range[n][1] ] )
        testin.append(  features[0][n][ test_range[n][0]:test_range[n][1] ] )
        testout.append(  cl_targets[n][ test_range[n][0]:test_range[n][1] ] )
    
    return trainin, trainout, testin, testout

def multiclass_dataset(train_files,test_files,
                        label=0.9,bias=1.,
                        scale_min=0., scale_max=1.,scale_prop="local",
                        feature_key="features",target_key="target_midi"):
    """ Multiclass dataset for anytime classification (e.g. for melody extraction)
    with multiple data files.
    
    train_files   --   list with files for training
    test_files    --   list with files for testing
    label         --   value for the target output (class1 is +label,
                       class2 is -label)
    bias          --   value of the bias input
    scale_min     --   minimum of each channel after scaling
    scale_max     --   maximum of each channel after scaling
    scale_prop    --   different scaling methods:
                       "local"   --    scales each channel to its min/max
                       "global"  --    scales each channel to the global min/max
                       "noscale" --    no scaling at all
    feature_key   --   key for shelve data, to get the features
    target_key    --   key for shelve data, to get the targets
    """
    # read all features
    features = []
    targets = []
    feature = []
    target = []
    for file in train_files:
        data = shelve.open(file)
        print file,"feature shape:", data[feature_key].shape
        feature.append(data[feature_key])
        target.append(data[target_key])
        data.close()
    features.append(feature)
    targets.append(target)
    feature = []
    target = []
    for file in test_files:
        data = shelve.open(file)
        print file,"feature shape:", data[feature_key].shape
        feature.append(data[feature_key])
        target.append(data[target_key])
        data.close()
    features.append(feature)
    targets.append(target)
    
    # make data preprocessing
    data_preprocessing(features,bias,scale_min,scale_max,0,scale_prop)

    # make targets
    
    # check how many pitch classes we have
    all_keys = []
    for el in targets[0]:
        all_keys += el.tolist()
    for el in targets[1]:
        all_keys += el.tolist()
    classes = list(set(all_keys))
    classes.sort()
    print "classes:", classes
    print "nr classes:",len(classes)
    
    # make (binary) target data
    cl_targets = []
    targ = []
    for piece in targets[0]:
        target = np.ones((len(piece), len(classes))) * (-1)*label
        for n in range(len(piece)):
            ind = classes.index( piece[n] )
            target[n,ind] = label
        targ.append(target)
    cl_targets.append(targ)
    targ = []
    for piece in targets[1]:
        target = np.ones((len(piece), len(classes))) * (-1)*label
        for n in range(len(piece)):
            ind = classes.index( piece[n] )
            target[n,ind] = label
        targ.append(target)
    cl_targets.append(targ)
    
    # make train and test data
    trainin = features[0]
    testin = features[1]
    trainout = cl_targets[0]
    testout = cl_targets[1]

    return trainin, trainout, testin, testout

def multiclass_dataset_enc(train_files,test_files,
                           label=0.9,bias=1.,
                           scale_min=0., scale_max=1.,scale_prop="local",
                           feature_key="features",target_key="target_chromas"):
    """ same as "multiclass_dataset, but for already encoded target data.
    
    train_files   --   list with files for training
    test_files    --   list with files for testing
    label         --   value for the target output (class1 is +label,
                       class2 is -label)
    bias          --   value of the bias input
    scale_min     --   minimum of each channel after scaling
    scale_max     --   maximum of each channel after scaling
    scale_prop    --   different scaling methods:
                       "local"   --    scales each channel to its min/max
                       "global"  --    scales each channel to the global min/max
                       "noscale" --    no scaling at all
    feature_key   --   key for shelve data, to get the features
    target_key    --   key for shelve data, to get the targets
    """
    # read all features
    features = []
    targets = []
    feature = []
    target = []
    for file in train_files:
        data = shelve.open(file)
        print file,"feature shape:", data[feature_key].shape
        feature.append(data[feature_key])
        target.append(data[target_key])
        data.close()
    features.append(feature)
    targets.append(target)
    feature = []
    target = []
    for file in test_files:
        data = shelve.open(file)
        print file,"feature shape:", data[feature_key].shape
        feature.append(data[feature_key])
        target.append(data[target_key])
        data.close()
    features.append(feature)
    targets.append(target)
    
    # make data preprocessing
    data_preprocessing(features,bias,scale_min,scale_max,0,scale_prop)

    # make targets
    mins,maxs = calc_minmax(targets[0])
    mins = mins.min()
    maxs = maxs.max()
    for m in range(2):
        for n in range(len(targets[0])):
            targets[m][n][np.where(targets[m][n] == mins)] = (-1)*label
            targets[m][n][np.where(targets[m][n] == maxs)] = label
    
    # make train and test data
    trainin = features[0]
    testin = features[1]
    trainout = targets[0]
    testout = targets[1]

    return trainin, trainout, testin, testout
    

################
# TEST
#datafiles = ["/home/holzi/phd/python/MelodyExtraction/data/ATasteOfHoney_STFT.dat",
#             "/home/holzi/phd/python/MelodyExtraction/data/Help_STFT.dat"]
#trainrange = [ [0,1000], [0,900] ]
#testrange = [ [1000,1800], [900,1600] ]
#
#trainin, trainout, testin, testout = multiclass_anytime2(datafiles, trainrange, testrange)
#print trainin[0].shape, testin[0].shape
#print trainout[0].shape, testout[0].shape
#print testout[0][100:200]
