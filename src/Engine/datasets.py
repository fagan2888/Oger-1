'''
Created on Aug 20, 2009

@author: dvrstrae
'''
from Engine import mdp
import glob
import os
import collections
import scipy as sp
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

def heating_tank(sample_len=1000, inputs=None):
    ''' Return data for the heating tank task. This system models the 
    temperature of the exhaust of a tank which is subject to a constant source 
    of heat, and where the control is the inflow of a source of cold liquid.
    This task is rather difficult since there is a substantial caused by the 
    exhaust.
    The system is described by the following differential equation:
    dy = (Q + rho*cp*q*(Tin-y))/(rho*cp*V_ves);
    
    where ...
    
    Based on a Matlab script by Tim Waegeman.
    '''
    
    if inputs is None:
        inputs = sp.sin(arange(0, sample_len / 10., 0.1))
    
    # Define some constants of the system
    Q = 1100
    rho = 1
    V_ves = 1.13
    Tin = 15
    cp = 4186
    V_tube = 1.02
    tautube = 29
    Ktube = 0.99
    
    T_tank = sp.array([])
    U = sp.array([])
    Y = sp.array([])
    
    # Sampling period
    Ts = 4
    
    # Equilibrium point
    q_star = 0.0167                      
    #Ttank_star = (Q / (q_star * rho * cp)) + Tin 
    #Ttube_star = Ktube * Ttank_star
    
    #for _ in range(200):

    delta_t = 0.1; # delta_t: time step
    timesteps = range(0, Ts, delta_t); # init time variable
 
    Ttankp = sp.zeros(timesteps)
    Ttankp[0] = T_tank[-1]; # init condition
        
    q = U[-1];
    
    for timestep in timesteps: 
        k1 = delta_t * (Q + rho * cp * q * (Tin - Ttankp[timestep])) / (rho * cp * V_ves) 
        k2 = delta_t * (Q + rho * cp * q * (Tin - (Ttankp[timestep] + (k1) / 2))) / (rho * cp * V_ves) 
        k3 = delta_t * (Q + rho * cp * q * (Tin - (Ttankp[timestep] + (k2) / 2))) / (rho * cp * V_ves) 
        k4 = delta_t * (Q + rho * cp * q * (Tin - (Ttankp[timestep] + (k3)))) / (rho * cp * V_ves) 

        Ttankp[timestep + 1] = Ttankp[timestep] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    T_tank = sp.insert(Ttankp[2:], 0, T_tank)
    print 'Warning: add lowpass!!'
    T_tube = Ktube * T_tank[0]#lowpass(T_tank, tautube / delta_t); # Apply filter           
    
    Tout = T_tube(-sp.floor((V_tube) / (inputs * Ts)));
    U = sp.insert(U, 0, input) # set new throughput   
    Y = sp.insert(Y, 0, Tout) # set new output
    return U, Y, T_tank

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
