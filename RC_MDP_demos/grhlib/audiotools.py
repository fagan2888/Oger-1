import numpy as np


def freq_to_midi(freq):
    """ Converts a frequency (Hz) into a MIDI note nr.
    
    freq   --   numpy array with frequency content
    """
    midi = 69 + 12*np.log(freq/440.)/np.log(2.)
    midi[np.nonzero(midi<0.)[0]] = 0. # check for -inf
    return midi

def midi_to_freq(midi):
    """ Converts a MIDI note nr into a frequency (Hz).
    
    midi   --   numpy array with midi note numbers
    """
    freq = 440 * 2**((midi-69)/12.)
    return freq

def buffering(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis into overlapping frames.

    Parameters:
    -----------
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is 
    unevenly strided and being flattened or because end is set to 
    'pad' or 'wrap').
    
    Example:
    --------
    >>> buffer(np.arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError, "frames cannot overlap by more than 100%"
    if overlap<0 or length<=0:
        raise ValueError, "overlap must be nonnegative and length must be positive"

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = np.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b
        
        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError, "Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'"
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis]+(n,length)+a.shape[axis+1:]
    newstrides = a.strides[:axis]+((length-overlap)*s,s) + a.strides[axis+1:]

    try: 
        return np.ndarray.__new__(np.ndarray,strides=newstrides,shape=newshape,buffer=a,dtype=a.dtype)
    except TypeError:
        #warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis]+((length-overlap)*s,s) + a.strides[axis+1:]
        return np.ndarray.__new__(np.ndarray,strides=newstrides,shape=newshape,buffer=a,dtype=a.dtype)

def unbuffering(a, overlap=0, window=None):
    """Reconstructs the original array, which was segmented in frames
    with the function "buffer".
    
    Parameters:
    -----------
    a       The array with frames
    overlap The number of array elements by which the frames overlap
    window  window function for overlapping frames (e.g. scipy.signal.hann)

    The array is copied if the overlap is not zero !

    TODO: multidimensional arrays !
    """
    
    # no overlap - no copy
    if(overlap==0):
        return a.reshape(a.size)
    
    if(window==None):
        raise ValueError, "Please specify a window function !"
    
    # calc original data length
    frames = a.shape[0]
    framesize = a.shape[1]
    olfactor = float(framesize) / float(framesize-overlap)
    origlength = np.ceil(frames/olfactor) * framesize + overlap
    
    # perform the overlap and add
    newdata = np.zeros(origlength)
    for n in range(frames):
        istart = n*(framesize-overlap)
        iend = istart+framesize
        newdata[istart:iend] += a[n] * window(framesize)

    return newdata

def cartesian2polar(n):
    """ Converts the complex number n to polar coordinates (abs, angle).
    """
    return np.abs(n), np.angle(n)

def polar2cartesian(ampl, phase):
    """ Converts from polar coordinates to one cartesian complex number.
    
    If n is a matrix ,then angle and phase is calculated in rows
    (to be used with buffering/unbuffering).
    """
    # TODO: machen dass er nicht automatisch 128 bit ist !
    cart = np.asfarray( (ampl * np.cos(phase)), 'complex128')
    cart.imag = (ampl * np.sin(phase))
    return cart


#===============================================================================
# MAIN function
#===============================================================================
if __name__ == "__main__":

    # lets test the buffering/unbuffering
#    signal = np.arange(10)
#    print "input signal: ", signal
#    
#    frames = buffering(signal, 2)
#    print "frames with framelength 2 and no overlap: ", frames
#    
#    orig = unbuffering(frames)
#    print "reconstruction of the original signal: ", orig
    
    # MIDI test
    midi = np.array([59., 12, 68.])
    freq = midi_to_freq(midi)
    midi_new = freq_to_midi(freq)
    print midi, freq, midi_new
    
