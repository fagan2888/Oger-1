import numpy as np

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
