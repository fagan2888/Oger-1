import numpy as np
import mdp

def nrmse(input_signal, target_signal):
    """ NRMSE calculation.
    
    Calculates the normalized root mean square error (NRMSE)
    of the input signal compared to the target signal.
    
    Parameters:
    -----------
        input_signal : array
            the input_signal signal
        target_signal : array
            the target signal
    """
    input_signal = input_signal.flatten()
    target_signal = target_signal.flatten()
    
    if(target_signal.size != input_signal.size):
        raise RuntimeError('Length of target signal is not equal to length of generated signal.')
    
    # Use normalization with N-1, as in matlab
    var = target_signal.std(ddof=1) ** 2
    
    error = (target_signal - input_signal) ** 2
    return np.sqrt(error.mean() / var)
    
def nmse(input_signal, target_signal):
    """ NMSE calculation.
    
    Calculates the normalized mean square error (NMSE)
    of the input signal compared to the target signal.
    
    Parameters:
    -----------
        input_signal : array
            the input signal
        target_signal : array
            the target signal
    """
    input_signal = input_signal.flatten()
    targetsignal = target_signal.flatten()
        
    if(targetsignal.size != input_signal.size):
        raise RuntimeError('Length of target_signal signal is not equal to length of generated signal.')
    
    var = targetsignal.std()**2
    
    error = (targetsignal - input_signal) ** 2
    return error.mean() / var
    
def rmse(input_signal, target_signal):
    """ RMSE calculation.
    
    Calculates the root mean square error (RMSE)
    of the input signal compared target target signal.
    
    Parameters:
    -----------
        input_signal : array
            the input signal
        target_signal : array
            the target signal
    """
  
    error = (target_signal.flatten() - input_signal.flatten()) ** 2
    return np.sqrt(error.mean())
    
def mse(input_signal, target_signal):
    """ MSE calculation.
    
    Calculates the mean square error (MSE)
    of the input signal compared target signal.
    
    Parameters:
    -----------
        input_signal : array
            the input_signal signal
        targetsignal : array
            the target_signal signal
    """            
    error = (targetsignal.flatten() - input_signal.flatten()) ** 2
    return error.mean()

def loss_01(input_signal, target_signal):
    """ Compute zero one loss function 
    
    Returns the fraction of timesteps where input_signal is unequal to target_signal
    
    Parameters:
    -----------
        input_signal : array
            the input_signal signal
        target_signal : array
            the target_signal signal
    """
    if input_signal.shape != target_signal.shape:
        raise RuntimeError("Input and target_signal should have the same shape")

    return (input_signal.flatten() != target_signal.flatten()).mean()

def cosine(input_signal, target_signal):
    ''' Compute cosine of the angle between two vectors
    
    This error measure measures the extent to which two vectors point in the same direction. 
    A value of 1 means complete alignment, a value of 0 means the vectors are orthogonal.
    '''
    return float(mdp.numx.dot(input_signal, target_signal)) / (mdp.numx.linalg.norm(input_signal) * mdp.numx.linalg.norm(target_signal))
