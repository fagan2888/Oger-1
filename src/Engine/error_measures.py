from __future__ import division
import numpy as np
import mdp

__all__ = [
    'nrmse',
    'nmse',
    'rmse',
    'mse'
    ]

def nrmse( input, target, discard=0, var=-1 ):
    """ NRMSE calculation.
    
    Calculates the normalized root mean square error (NRMSE)
    of the input signal compared to the target signal.
    
    Parameters:
    -----------
        input : array
            the input signal
        target : array
            the target signal
        discard : int, optional
            number of initial values which should be skipped
        var : float, optional
            can be used to set the variance of the target signal
    """
    insignal = input.copy()
    targetsignal = target.copy()
    
    # reshape values
    insignal.shape = -1,
    targetsignal.shape = -1,
    
    if( targetsignal.size != insignal.size ):
        raise RuntimeError('Length of target signal is not equal to length of generated signal.')
    
    # check if a variance is given
    if var<=0:
        var = targetsignal.std()**2
    
    error = (targetsignal - insignal)**2
    nrmse = np.sqrt( error.mean() / var )
    
    return nrmse

def nmse( input, target, discard=0, var=-1 ):
    """ NMSE calculation.
    
    Calculates the normalized mean square error (NMSE)
    of the input signal compared to the target signal.
    
    Parameters:
    -----------
        insignal : array
            the input signal
        targetsignal : array
            the target signal
        discard : int, optional
            number of initial values which should be skipped
        var : float, optional
            can be used to set the variance of the target signal
    """
    insignal = input.copy()
    targetsignal = target.copy()
    
    # reshape values
    insignal.shape = -1,
    targetsignal.shape = -1,
    
    if( targetsignal.size > insignal.size ):
        maxsize = insignal.size
    else:
        maxsize = targetsignal.size
    
    origsig = targetsignal[discard:maxsize]
    testsig = insignal[discard:maxsize]
    
    # check if a variance is given
    if var<=0:
        var = origsig.std()**2
    
    error = (origsig - testsig)**2
    nmse = error.mean() / var
    
    return nmse

def rmse( input, target, discard=0 ):
    """ RMSE calculation.
    
    Calculates the root mean square error (RMSE)
    of the input signal compared to the target signal.
    
    Parameters:
    -----------
        insignal : array
            the input signal
        targetsignal : array
            the target signal
        discard : int, optional
            number of initial values which should be skipped
    """
    insignal = input.copy()
    targetsignal = target.copy()
    
    # reshape values
    insignal.shape = -1,
    targetsignal.shape = -1,
    
    if( targetsignal.size > insignal.size ):
        maxsize = insignal.size
    else:
        maxsize = targetsignal.size
    
    origsig = targetsignal[discard:maxsize]
    testsig = insignal[discard:maxsize]
    
    error = (origsig - testsig)**2
    nrmse = np.sqrt( error.mean() )
    
    return nrmse

def mse( input, target, discard=0 ):
    """ MSE calculation.
    
    Calculates the mean square error (MSE)
    of the input signal compared to the target signal.
    
    Parameters:
    -----------
        insignal : array
            the input signal
        targetsignal : array
            the target signal
        discard : int, optional
            number of initial values which should be skipped
    """
    insignal = input.copy()
    targetsignal = target.copy()
    
    # reshape values
    insignal.shape = -1,
    targetsignal.shape = -1,
    
    if( targetsignal.size > insignal.size ):
        maxsize = insignal.size
    else:
        maxsize = targetsignal.size
    
    origsig = targetsignal[discard:maxsize]
    testsig = insignal[discard:maxsize]
    
    error = (origsig - testsig)**2
    nrmse = error.mean()
    
    return nrmse

def zero_one_loss(input, target, discard=0):
    """ Compute mean zero-one loss function
    
    This error measure is the zero-one loss:
    - if the input signal is not equal to the desired signal, one is added to the error
    - if the input signal is equal to the desired signal, nothing is added.
    This error is then averaged over all timesteps
    
    
    Parameters:
    -----------
        input : array
            the input signal
        target : array
            the target signal
        discard : int, optional
            number of initial values which should be skipped
    """
    insignal = input.copy()
    targetsignal = target.copy()
    
    # reshape values
    insignal.shape = -1,
    targetsignal.shape = -1,
    
    if( targetsignal.size > insignal.size ):
        maxsize = insignal.size
    else:
        maxsize = targetsignal.size
    
    targetsignal = targetsignal[discard:maxsize]
    insignal = insignal[discard:maxsize]
    
    print insignal!=targetsignal
    
    error = np.sum(insignal!=targetsignal)
    print error
    zero_one_loss = error.mean()
    
    return zero_one_loss

def loss_01(input, target, discard=0):
    """ Compute word error rate
    
    This error measure is the word error rate:
    - class labels are assumed to be -1 and +1
    - If all class labels are equal, error is zero
    - If some class labels are not equal, error is one.
    Input and target are assumed to be vectors (single timestep).
    
    Parameters:
    -----------
        input : array
            the input signal
        target : array
            the target signal
        discard : int, optional
            number of initial values which should be skipped
    """
    if input.shape!=target.shape:
        raise RuntimeError("Input and target should have the same shape")
    #print input, target
    return sum(input[:] !=target[:])    

def cosine(x, y):
    ''' Compute cosine of angle between two vector
    
    This error measure measures the extent to which two vectors point in the same direction. 
    A value of 1 means complete alignment, a value of 0 means the vectors are orthogonal.
    '''
    return mdp.numx.dot(x, y) / (mdp.numx.linalg.norm(x) * mdp.numx.linalg.norm(y))
