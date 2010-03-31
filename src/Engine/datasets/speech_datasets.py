'''
Created on Mar 31, 2010

@author: bschrauw
'''
import mdp
import glob
import os
from scipy.io import loadmat

def analog_speech (indir='../datasets/Lyon128'):
    ''' Return data for the isolated digit recognition task (subset of TI46), 
        preprocessed using the Lyon Passive ear model.
    '''
    speech_files = glob.glob(os.path.join(indir, '*.mat'))
    inputs, outputs = [], []
    if len(speech_files) > 0:
        print "Found %d speech_files in directory %s. Loading..." % \
            (len(speech_files), indir)
        #for speech_file in mdp.utils.progressinfo(speech_files):
        for speech_file in speech_files:
            contents = loadmat(speech_file)
            inputs.append(contents['spec'].T)
            outputs.append(-1 * mdp.numx.ones([inputs[-1].shape[0], 10]))
            # Fourth last character in filename indicates the digit class
            outputs[-1][:, speech_file[-5]] = 1           
    else:
        print "No speech_files found in %s" % (indir)
        return
    return inputs, outputs

def timit_tiny (indir='/data/aurora/Variables/TIMIT/timittiny'):
    ''' Return data for a small subset of the TIMIT dataset
    '''
    timit_files = glob.glob(os.path.join(indir, '*.mat'))
    inputs, outputs = [], []
    if len(timit_files) > 0:
        print "Found %d files in directory %s. Loading..." % \
        (len(timit_files), indir)
        for timit_file in mdp.utils.progressinfo(timit_files):
            contents = loadmat(timit_file)
            inputs.append(contents['spec'].T)
        outputs.append(contents['target'])
    else:
        print "No files found in %s" % (indir)
        return
    return inputs, outputs
