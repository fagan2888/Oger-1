#===============================================================================
# file with some test signals
#===============================================================================

from numpy import *
import pylab


def sine(size, freq, amp=1., phase=0.):
    """ generates a sines with the given frequency and amplitude
    """
    x = arange( float(size) )
    x = amp * sin( 2*pi*x* freq + phase )
    return x

def am_sine(size, freq, mod_freq, amp=1., mod_amp=1.):
    """ amplitude modulation of a sine
    """
    ind = arange( float(size) )
    mod = mod_amp * sin( 2*pi*ind* mod_freq )
    x = amp * mod * sin( 2*pi*ind* freq )
    return x, mod

def fm_sine(size, freq, mod_freq, amp=1., mod_amp=2.5):
    """ phase modulation of a sine
    """
    ind = arange( float(size) )
    phase_mod = mod_amp * sin( 2*pi*ind* mod_freq )
    x = amp * sin( 2*pi*ind* freq + phase_mod )
    return x, phase_mod

def randwalk_sine(size, freq, amp=1., walk_size=0.1, walk_boundary=3.):
    """ phase modulation of a sine with random walk
    """
    ind = arange( float(size) )
    
    # random walk signal
    phase = zeros(size)
    for n in range(1,size):
        step = (random.rand(1)-0.5) * walk_size
        phase[n] = phase[n-1] + step
        if abs(phase[n]) > walk_boundary:
            phase[n] = phase[n] - 2*step
    
    x = amp * sin( 2*pi*ind* freq + phase )
    return x, phase

def fm_sine_special(size, freq, amp=1., mod_ampl=2):
    """ phase modulation of a sine
    """
    ind = arange( float(size) )
    t = linspace(0,1,size,endpoint=0) # time axis 1s, samplerate 10KHz
    phase_mod = sin(2*pi*5*t) + sin(2*pi*11*t) + sin(2*pi*13*t)
    x = amp * sin( 2*pi*ind* freq + mod_ampl * phase_mod )
    return x, phase_mod

def rectangular(size, freq, amp=1.):
    """ generates a rectangular wave with the given frequency and amplitude
    """
    x = arange( float(size) )
    x = sin( 2*pi*x* freq )
    x = amp * sign(x)
    return x

def triangular(size, freq, amp=1.):
    """ generates a triangular wave with the given frequency and amplitude
    """
    T = 1. / freq
    
    # make one period
    steps = int(T)
    snip = zeros(steps)
    for n in range(steps):
        snip[n] = float(n) / float(steps)
    snip = 2*snip - 1
    
    # make the whole signal
    nr = int(ceil( float(size) / steps ))
    x = amp * tile(snip, nr)
    return x[0:size]

def add_noise(signal, noiselevel):
    """ adds noise to a signal
    """
    noise = (random.rand(len(signal))*2.-1.) * noiselevel
    signal += noise

#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    """ some plotting for tests
    """
    
#    x1 = sine(1000, 0.01, 1)
#    x2 = rectangular(1000, 0.01, 0.5)
#    x3 = triangular(1000,0.01, 0.8)
    
#    am, mod = am_sine(10000, 0.01, 0.0005)
    fm, mod = fm_sine(10000, 0.005, 0.001, 1, 10)
    
#    pylab.psd(fm)
    
    pylab.subplot(211)
    pylab.plot( fm )
    pylab.subplot(212)
    pylab.plot( mod )
    pylab.show()
