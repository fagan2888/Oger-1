import numpy as np
import mdp

def nrmse(input, target):
    """ NRMSE calculation.
    
    Calculates the normalized root mean square error (NRMSE)
    of the input signal compared to the target signal.
    
    Parameters:
    -----------
        input : array
            the input signal
        target : array
            the target signal
    """
    input = input.copy()
    targetsignal = target.copy()
    
    # reshape values
    input.shape = -1,
    targetsignal.shape = -1,
    
    if( targetsignal.size != input.size ):
        raise RuntimeError('Length of target signal is not equal to length of generated signal.')
    
    # Use normalization with N-1, as in matlab
    var = targetsignal.std(ddof=1)**2
    
    error = (targetsignal - input)**2
    return np.sqrt( error.mean() / var )
    
def nmse(input, target):
    """ NMSE calculation.
    
    Calculates the normalized mean square error (NMSE)
    of the input signal compared to the target signal.
    
    Parameters:
    -----------
        input : array
            the input signal
        targetsignal : array
            the target signal
    """
    input = input.copy()
    targetsignal = target.copy()
    
    # reshape values
    input.shape = -1,
    targetsignal.shape = -1,
    
    if( targetsignal.size != input.size ):
        raise RuntimeError('Length of target signal is not equal to length of generated signal.')
    
    var = targetsignal.std()**2
    
    error = (targetsignal - input)**2
    return error.mean() / var
    
def rmse(input, target):
    """ RMSE calculation.
    
    Calculates the root mean square error (RMSE)
    of the input signal compared to the target signal.
    
    Parameters:
    -----------
        input : array
            the input signal
        targetsignal : array
            the target signal
    """
    input = input.copy()
    targetsignal = target.copy()
    
    # reshape values
    input.shape = -1,
    targetsignal.shape = -1,
            
    error = (targetsignal - input)**2
    return np.sqrt( error.mean() )
    
def mse(input, target):
    """ MSE calculation.
    
    Calculates the mean square error (MSE)
    of the input signal compared to the target signal.
    
    Parameters:
    -----------
        input : array
            the input signal
        targetsignal : array
            the target signal
    """
    
    input = input.copy()
    targetsignal = target.copy()
    
    # reshape values
    input.shape = -1,
    targetsignal.shape = -1,
        
    error = (targetsignal - input)**2
    return error.mean()

def loss_01(input, target):
    """ Compute zero one loss function 
    
    Returns the fraction of timesteps where input is unequal to target
    
    Parameters:
    -----------
        input : array
            the input signal
        target : array
            the target signal
    """
    if input.shape!=target.shape:
        raise RuntimeError("Input and target should have the same shape")

    return (input[:] !=target[:]).mean()

def cosine(input, target):
    ''' Compute cosine of the angle between two vectors
    
    This error measure measures the extent to which two vectors point in the same direction. 
    A value of 1 means complete alignment, a value of 0 means the vectors are orthogonal.
    '''
    return float(mdp.numx.dot(input, target)) / (mdp.numx.linalg.norm(input) * mdp.numx.linalg.norm(target))

def ce(input, target):
    """ Compute cross-entropy loss function

    Returns the negative log-likelyhood of the target labels as predicted by
    the input values.

    Parameters:
    -----------
        input : array
            the input signal
        target : array
            the target signal
    """
    if input.shape!=target.shape:
        raise RuntimeError("Input and target should have the same shape")

    return np.sum(-np.log(input[target == 1]))

