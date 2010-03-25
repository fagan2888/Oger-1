'''
Created on Aug 20, 2009

@author: dvrstrae
'''
import mdp
import glob
import os
import collections
from scipy.io import loadmat

def narma30(n_samples=10, sample_len=1000):
    ''' Return data for the 30th order NARMA task.  
    '''
    system_order = 30
    inputs, outputs = [], []
    for sample in range(n_samples):
        inputs.append(mdp.numx.random.rand(sample_len, 1) * .5)
        inputs[sample].shape = (-1, 1)
        outputs.append(mdp.numx.zeros((sample_len, 1)))
        for k in range(system_order, sample_len - 1):
            outputs[sample][k + 1] = .2 * outputs[sample][k] + .04 * \
            outputs[sample][k] * mdp.numx.sum(outputs[sample][k - 29:k]) + \
            1.5 * inputs[sample][k - 29] * inputs[sample][k] + .001
    return inputs, outputs 

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

def mackey_glass(sample_len=1000, tau=17):
    ''' Generate the Mackey Glass time-series. Parameters are:
        - sample_len: length of the time-series in timesteps
        - tau: delay of the MG - system. Commonly used values are tau=17 (mild 
          chaos) and tau=30 (moderate chaos) 
    '''
    delta_t = 10
    history_len = tau * delta_t 
    # Initial conditions for the history of the system
    timeseries = 1.2
    history = collections.deque(1.2 * mdp.numx.ones(history_len) + 0.2 * \
                                (mdp.numx.random.rand(history_len) - 0.5))
    # Preallocate the array for the time-series
    inp = mdp.numx.zeros([sample_len])
    
    for timestep in range(sample_len):
        for _ in range(delta_t):
            xtau = history.popleft()
            history.append(timeseries)
            timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                         0.1 * history[-1]) / delta_t
        inp[timestep] = timeseries
    
    # Squash timeseries through tanh
    inp = mdp.numx.tanh(inp - 1)
    inp.shape = (-1, 1)
    return inp

def mso(sample_len=1000):
    x = mdp.numx.arange(0, sample_len, 1)
    x.shape += (1,)
    signal = mdp.numx.sin(0.2 * x) + mdp.numx.sin(0.311 * x) 
    return [signal, ]

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
