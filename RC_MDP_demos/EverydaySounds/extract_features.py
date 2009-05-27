#===============================================================================
# Python MFCC feature extraction
#===============================================================================

from scipy.io import wavfile
import fileinput
import sys
import shelve
import datasets
sys.path.append("../grhlib")
from mfcc import MFCC

# path to the audio files
DATA_PATH= datasets.CLOSED_AUDIO_PATH

# path with example lists (classes)
LIST_PATH = datasets.CLOSED_LIST_PATH

# all classes
SOUND_LISTS = datasets.SOUND_LISTS

# output directory for feature data
OUTPUT_DIR = 'data/'



def extract_mfccs(filename, framerate=100):
    """ Extracts MFCCs of one audio file.
    """
    print "Extracting MFCCs of", filename
    
    data = wavfile.read(filename)
    srate = data[0]
    samples = data[1]
    
    # setup MFCC class (as slaneys toolbox)
    mfcc_extr = MFCC(nfilt=40, ncep=13,
                     lowerf=133.3333, upperf=5055.4976, alpha=0.97,
                     samprate=srate, frate=framerate, winsize=256,
                     nfft=512)
    
    # extract features
    features = mfcc_extr.sig2s2mfc(samples)
    
    return features

def process_sound(example='flow.list'):
    """ Calculate feature for one sound category.
    """
    files = []
    for line in fileinput.input(LIST_PATH+example):
        if len(line)>3: # to ignore newlines
            files.append(DATA_PATH+line[:-1]) # remove newline
    
    # list with features of all files
    features = []
    for file in files:
        features.append( extract_mfccs(file) )
    
    # make the label
    label = example[:-5]
    
    # save data to disk
    savefile = OUTPUT_DIR+label+"_mfcc_fr100.dat"
    data = shelve.open(savefile)
    data["features"] = features
    data["label"] = label
    data.close()
    print "\nWritten features to", savefile


#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    # process one
    process_sound('flow.list')
    
    # process all
#    for example in SOUND_LISTS:
#        process_sound(example)
