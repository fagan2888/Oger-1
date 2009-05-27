#===============================================================================
# Python MFCC feature extraction
# (implements the same as the matlab script,
#  but with the python mfcc implementation)
#===============================================================================

from scipy.io import wavfile
import numpy as np
import fileinput
import sys
import shelve
sys.path.append("../../grhlib")
from mfcc import MFCC

# path to the audio files
DATA_PATH= '/home/holzi/Datasets/closed/data/audio_library/audio_files/'

# path with example lists
LIST_PATH = '/home/holzi/Datasets/closed/data/general_everyday_sounds/lists/'

# output directory for feature data
OUTPUT_DIR = '../data/'

SOUND_LISTS = ["deformation.list", "explosion.list", "friction.list", "pour.list",
               "whoosh.list", "drip.list", "flow.list", "impact.list",
               "rolling.list", "wind.list"]



def extract_mfccs(filename):
    """ Extracts MFCCs of one audio file.
    """
    print "Extracting MFCCs of", filename
    
    data = wavfile.read(filename)
    srate = data[0]
    samples = data[1]
    
    # setup MFCC class (as slaneys toolbox)
    mfcc_extr = MFCC(nfilt=40, ncep=13,
                     lowerf=133.3333, upperf=5055.4976, alpha=0.97,
                     samprate=srate, frate=100, winsize=256,
                     nfft=512)
    
    # extract features
    features = mfcc_extr.sig2s2mfc(samples)
    return features

def feature_integration(filename):
    """ Calculates mean, std and delta features of the whole file.
    """
    features = extract_mfccs(filename)

    # calc delta features
    features_delta = np.diff(features,axis=0) # 1st-order difference between the rows

    # combine the whole feature vector
    combined = np.r_[ features.mean(0),
                      features.std(0),
                      features_delta.mean(0),
                      features_delta.std(0) ]
    
    return combined

def process_sound(example='flow.list'):
    """ Calculate feature for one sound category.
    """
    files = []
    for line in fileinput.input(LIST_PATH+example):
        if len(line)>3: # to ignore newlines
            files.append(DATA_PATH+line[:-1]) # remove newline
    
    # allocate space for the features of each file
    features = np.zeros((len(files),52))
    
    for n in range(len(files)):
    #for n in range(1):
        features[n] = feature_integration(files[n])
    
    # save data to disk
    savefile = OUTPUT_DIR+example[:-5]+"_mfccint.dat"
    data = shelve.open(savefile)
    data["features"] = features
    data.close()
    print "\nWritten features to", savefile


#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    # process one
#    process_sound('flow.list')
    
    # process all
    for example in SOUND_LISTS[8:]:
        process_sound(example)
