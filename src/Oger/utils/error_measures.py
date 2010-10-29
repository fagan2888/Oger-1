import mdp
import numpy as np

def timeslice(range, function):
    """
    timeslice(range, function) -> function
    Apply the given function only to the given time range of the data.
    Can be used to eg. apply an error metric only to a part of the data.
    """
    return lambda x, y: function(x[range, :], y[range, :])
    
def nrmse(input_signal, target_signal):
    """ 
    nrmse(input_signal, target_signal)-> error
    NRMSE calculation.
    
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
    return mdp.numx.sqrt(error.mean() / var)
    
def nmse(input_signal, target_signal):
    """ 
    nmse(input_signal, target_signal)->error
    NMSE calculation.
    
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
    """
    rmse(input_signal, target_signal)->error 
    RMSE calculation.
    
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
    return mdp.numx.sqrt(error.mean())
    
def mse(input_signal, target_signal):
    """ 
    mse(input_signal, target_signal)->error
    MSE calculation.
    
    Calculates the mean square error (MSE)
    of the input signal compared target signal.
    
    Parameters:
    -----------
        input_signal : array
            the input_signal signal
        targetsignal : array
            the target_signal signal
    """   
    
    error = (target_signal.flatten() - input_signal.flatten()) ** 2
    return error.mean()

def loss_01(input_signal, target_signal):
    """ 
    loss_01(input_signal, target_signal)->error
    Compute zero one loss function 
    
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
    ''' 
    cosine(input_signal, target_signal)->cosine
    Compute cosine of the angle between two vectors
    
    This error measure measures the extent to which two vectors point in the same direction. 
    A value of 1 means complete alignment, a value of 0 means the vectors are orthogonal.
    '''
    return float(mdp.numx.dot(input_signal, target_signal)) / (mdp.numx.linalg.norm(input_signal) * mdp.numx.linalg.norm(target_signal))

def ce(input, target):
    """ 
    ce(input, target)-> cross-entropy
    Compute cross-entropy loss function

    Returns the negative log-likelyhood of the target labels as predicted by
    the input values.

    Parameters:
    -----------
        input : array
            the input signal
        target : array
            the target signal
    """
    if input.shape != target.shape:
        raise RuntimeError("Input and target should have the same shape")
    
    if target.shape[1]>1:
        error = mdp.numx.sum(-mdp.numx.log(input[target == 1]))
        
        if mdp.numx.isnan(error):
            inp = input[target == 1]
            inp[inp==0] = float(finfo(input.dtype).tiny)
            error = - mdp.numx.sum(mdp.numx.log(inp))
    else:
        error = - mdp.numx.sum(mdp.numx.log(input[target==1]))
        error -= mdp.numx.sum(mdp.numx.log(1 - input[target==0]))
        
        if mdp.numx.isnan(error):
            inp = input[target==1]
            inp[inp==0] = float(finfo(input.dtype).tiny)
            error = - mdp.numx.sum(mdp.numx.log(inp))
            inp = 1 - input[target==0]
            inp[inp==0] = float(finfo(input.dtype).tiny)
            error -= mdp.numx.sum(mdp.numx.log(inp))
    
    return error


# TODO: if we add container object for the error metrics, we should add a field that
# signifies if we need to minimize or maximize the measure
def mem_capacity(input, target):
    """Computes the memory capacity defined by Jaeger in 
    H. Jaeger (2001): Short term memory in echo state networks. GMD Report 
    152, German National Research Center for Information Technology, 2001
    
    WARNING: currently this returns the negative of the memory capacity so 
    we can keep on using the minimization code.
    """

    score = []
    for k in range(target.shape[1]):
        covariance_matrix = mdp.numx.cov(mdp.numx.concatenate((input[:, k:k + 1].T, target[:, k:k + 1].T)))
        score.append(covariance_matrix[0, 1] ** 2 / (covariance_matrix[0, 0] * covariance_matrix[1, 1]))
    
    return - mdp.numx.sum(score)


def threshold_before_error(input, target, error_measure, thresh=None):
    """
    First applies a threshold to input and target and then determines the error using the error_measure function.
    The threshold is estimated as the mean of the target maximum and minimum unless a threshold 'thresh' is specified
    
    Useful for classification error estimation. Example:
        error_measure = lambda x,y: Oger.utils.threshold_before_error(x, y, Oger.utils.loss_01)
    """
    if thresh == None:
        thresh = (max(target) + min(target)) / 2
    return error_measure(input>thresh, target>thresh)
    

def ber(input, target):
    """ 
    ber(input, target)->error
    Compute Balanced Error Rate 
    
    Returns the average of the fraction of timesteps where input_signal is unequal to target_signal for each class
    Only compatible with 2 classes (TODO multiclass)
    
    Parameters:
    -----------
        input : array
            the input signal
        target : array
            the target signal
    """
    (tp, fp, tn, fn) = _conf_table(input, target)
    
    return _ber(tp, fp, tn, fn)


def f_score(input, target, beta=1.0):
    """ 
    f_score(input, target, beta=1)->error
    Compute 1 minus F-beta score, for beta=1 it computes the 1 minus F1 score 
    
    Parameters:
    -----------
        input : array
            the input signal
        target : array
            the target signal
    """
    (tp, fp, tn, fn) = _conf_table(input, target)
    
    return _f_beta(tp, fp, tn, fn, beta)


def _conf_table(input, target):
    '''
    Helper function to determine the number of:
        True Positives
        False Positives
        True Negatives
        False Negatives
    '''
    if input.shape != target.shape:
        raise RuntimeError("Input and target_signal should have the same shape")
    
    tp = np.sum(np.logical_and(input.flatten()>0, target.flatten()>0))
    fp = np.sum(np.logical_and(input.flatten()>0, target.flatten()<=0))
    tn = np.sum(np.logical_and(input.flatten()<=0, target.flatten()<=0))
    fn = np.sum(np.logical_and(input.flatten()<=0, target.flatten()>0))
    return np.array([tp, fp, tn, fn])


def _ber(tp,fp,tn,fn):
    '''
    Helper function to determine the BER score
    Useful for direct estimation of the BER if the conf_table is known
    '''
    return 1.0 / 2 * (fp / (0.0 + tn + fp) + fn / (0.0 + fn + tp))


def _f_beta(tp, fp, tn, fn, beta=1.0):
    '''
    Helper function to determine the 1 minus F-beta score
    By default beta is 1 wich results in the F1 score
    Useful for direct estimation of the F-score if the conf_table is known
    '''
    return 1.0 - (1.0 + beta**2) * tp / ((1.0 + beta**2) * tp + beta**2 * fn + fp)

