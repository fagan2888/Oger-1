#===============================================================================
# convert various features to libsvm format
#===============================================================================

from scipy import io
import shelve
import random
from copy import deepcopy
import numpy as np

def write_featureset_to_file(FILE, features):
    """ Helper function.
    """
    for example in features:
        label = example[1]
        data = example[0]
        
        FILE.write(str(label)+" ")
        for n in range(data.shape[0]):
            FILE.write(str(n+1)+":")
            FILE.write(str(data[n])+" ")
        FILE.write("\n")

def combine_data(features1,features2, shuffle=0):
    """ Combines Data in random fashion.
    """
    label1 = +1
    label2 = -1
    
    features = []
    for file in features1:
        features.append((file,label1))
    for file in features2:
        features.append((file,label2))
    
    if shuffle:
        random.shuffle(features)
    return features
    

def make_matlab_1vs1(class1, class2, outfile, shuffle=0):
    """ Takes matlab mfcc features for 1vs1 classification and creates a
    file in libsvm format.
    """
    # read matlab data
    features1 = io.loadmat(class1)['features']
    features2 = io.loadmat(class2)['features']
    
    features = combine_data(features1,features2,shuffle)
    
    FILE = open(outfile,"w")
    write_featureset_to_file(FILE, features)
    FILE.close()

    print "Features written to ", outfile, " in libsvm Format!"

def make_matlab_1vsAll(class1, classRest, outfile, shuffle=0):
    """ Takes matlab mfcc features for 1vs1 classification and creates a
    file in libsvm format.
    """
    # read matlab data
    features1 = io.loadmat(class1)['features']
    
    featuresRest = np.zeros((0,52))
    for file in classRest:
        feat = io.loadmat(file)['features']
        featuresRest = np.r_[featuresRest, feat]

    features = combine_data(features1,featuresRest,shuffle)
    
    FILE = open(outfile,"w")
    write_featureset_to_file(FILE, features)
    FILE.close()

    print "Features written to ", outfile, " in libsvm Format!"

def make_python_1vs1(class1, class2, outfile, shuffle=0):
    """ Takes python mfcc features for 1vs1 classification and creates a
    file in libsvm format.
    """
    # read data
    data = shelve.open(class1)
    features1 = data["features"]
    data.close()
    data = shelve.open(class2)
    features2 = data["features"]
    data.close()
    
    features = combine_data(features1,features2,shuffle)
    
    FILE = open(outfile,"w")
    write_featureset_to_file(FILE, features)
    FILE.close()
    
    print "Features written to ", outfile, " in libsvm Format!"

def make_python_1vsAll(class1, classRest, outfile, shuffle=0):
    """ Takes matlab mfcc features for 1vs1 classification and creates a
    file in libsvm format.
    """
    # read data
    data = shelve.open(class1)
    features1 = data["features"]
    data.close()
    
    featuresRest = np.zeros((0,52))
    for file in classRest:
        data = shelve.open(file)
        feat = data["features"]
        data.close()
        featuresRest = np.r_[featuresRest, feat]

    features = combine_data(features1,featuresRest,shuffle)
    
    FILE = open(outfile,"w")
    write_featureset_to_file(FILE, features)
    FILE.close()

    print "Features written to ", outfile, " in libsvm Format!"

def generate_all_vsRest():
    """ Generates all 1vsRest datasets.
    """
    
    SOUND_LIST = ["../data/deformation_mfccint", "../data/explosion_mfccint",
                  "../data/friction_mfccint", "../data/pour_mfccint",
                  "../data/whoosh_mfccint", "../data/drip_mfccint",
                  "../data/flow_mfccint", "../data/impact_mfccint",
                  "../data/rolling_mfccint", "../data/wind_mfccint"]
    
    for n in range(len(SOUND_LIST)):
        all = deepcopy(SOUND_LIST)
        one = all.pop(n)

        # make matlab features
        outmat = one + '_vs_all_mat.libsvm'
        onemat = one + '.mat'
        allmat = []
        for file in all:
            allmat.append(file+'.mat')
        make_matlab_1vsAll(onemat,allmat,outmat)
        
        # make python features
        outdat = one + '_vs_all_dat.libsvm'
        onedat = one + '.dat'
        alldat = []
        for file in all:
            alldat.append(file+'.dat')
        make_python_1vsAll(onedat,alldat,outdat)        

#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    
    generate_all_vsRest()
    
#    make_matlab_1vs1("../data/drip_mfccint.mat",
#                     "../data/flow_mfccint.mat",
#                     "../data/drip_vs_flow_mat.libsvm")

#    make_python_1vs1("../data/drip_mfccint.dat",
#                     "../data/flow_mfccint.dat",
#                     "../data/drip_vs_flow_dat.libsvm")
