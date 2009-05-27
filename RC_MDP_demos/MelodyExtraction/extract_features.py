#===============================================================================
# Pre-Processing for the Melody Extraction Problem
#===============================================================================

import os
from scipy.io import loadmat, read_array, write_array
import numpy as np
import shelve
import glob
import pylab
import sys
sys.path.append("../grhlib")
import audiotools 

# path to the audio files
#DATA_PATH = "/home/holzi/Datasets/MelodyExtraction/Ismir04FullSet/"
DATA_PATH = "/home/holzi/Datasets/MelodyExtraction/mirex05TrainFiles/"
#DATA_PATH = "/home/holzi/phd/python/MelodyExtractionDataset/data/"

# directory for feature vector files
OUTPUT_DIR = "data/"


def normalized_stft(audiofile):
    """ Extracts STFT features as in the original paper by Graham Poliner.
    """
    mat_tmpfile = audiofile[:-4]+".mat"
    
    # perform the matlab feature extraction
    cmd = 'matlab -nosplah -nodisplay -nodesktop -nojvm -r \"preprocessData '
    cmd += DATA_PATH + audiofile + ' ' + mat_tmpfile + ';quit;\"'
    print "running Matlab:" 
    print cmd
    os.system(cmd)
    
    # read matlab file and save it to disk
    features = loadmat(mat_tmpfile)['data']
    savefile = OUTPUT_DIR + audiofile[:-4] + "_STFT.dat"
    data = shelve.open(savefile)
    data["features"] = features
    data["filename"] = audiofile
    data.close()
    
    # delete the matlab files
    os.remove(mat_tmpfile)

def transform_target_data_old(audiofile,analyze=0):
    """ Calculates the target data (in MIDI pitch classes) out of the REF.txt files.
    (for the old datasets of Ellis and MIREX)
    """
    target_file = DATA_PATH + audiofile[:-4] + "REF.txt"
    data = read_array(target_file)
    
    # remove time axes and convert to midi
    data = data[:,1]
    midi_data = audiotools.freq_to_midi(data)
    midi_data = np.round(midi_data)
    
    if analyze:
        pylab.plot(midi_data)
    
    # get the unique element (nr. of different notes) of the piece
    unique = list(set(midi_data))
    target = np.zeros((len(midi_data), len(unique)))
    for n in range( len(midi_data) ):
        ind = unique.index( midi_data[n] )
        target[n,ind] = 1
    
    if analyze:
        print "classes:",len(unique)
        pylab.figure()
        pylab.psd(target.flatten())
        pylab.show()
        exit(0)
    
    savefile = OUTPUT_DIR + audiofile[:-4] + "_STFT.dat"
    data = shelve.open(savefile)
    lenge = data["features"].shape[0]
    data["targets"] = target[:lenge]
    data["target_midi"] = midi_data[:lenge]
    data["pitch_classes"] = unique
    data.close()

def transform_target_data(audiofile,analyze=0):
    """ Gets the target data (in MIDI pitch classes) out of the REF.txt files.
    (for my own new datasets)
    """
    target_file = DATA_PATH + audiofile[:-4] + "REF.txt"
    data = read_array(target_file,',')
    
    if analyze:
        pylab.subplot(211)
        pylab.plot(data[:,1])
        pylab.subplot(212)
        pylab.plot(data[:,2])
        
        pylab.figure()
        pylab.subplot(211)
        pylab.hist(data[:,1],bins=25)
        pylab.xlim(40,65)
        pylab.subplot(212)
        pylab.hist(data[:,2],bins=13)
        pylab.xlim(0,12)
    
    # get the unique element (nr. of different notes) of the piece
    midi_data = data[:,1]
    unique = list(set(midi_data))
    target = np.zeros((len(midi_data), len(unique)))
    target_chromas = np.zeros((len(midi_data), 13))
    for n in range( len(midi_data) ):
        ind = unique.index( midi_data[n] )
        target[n,ind] = 1
        target_chromas[n,data[n,2]+1] = 1
    
    if analyze:
        print "pitch classes:",len(unique)
        pylab.show()
        exit(0)
    
    savefile = OUTPUT_DIR + audiofile[:-4] + "_STFT.dat"
    data = shelve.open(savefile)
    lenge = data["features"].shape[0]
    data["targets"] = target[:lenge]
    data["target_midi"] = midi_data[:lenge]
    data["target_chromas"] = target_chromas[:lenge]
    data["pitch_classes"] = unique
    data.close()

def convert_to_midi(reffile,savefile):
    """ converts REF files from mirex05 to MIDI data
    """
    target_file = DATA_PATH + reffile
    data = read_array(target_file)
    
    # remove time axes and convert to midi
    midi = np.round( audiotools.freq_to_midi(data[:,1]) )
    
    # calc chromas
    chromas = np.ones(len(midi)) * (-1)
    for n in range(len(midi)):
        if midi[n]>0:
            chromas[n] = midi[n] % 12
    
    # write data to disk
    writedata = np.c_[data[:,0], midi, chromas]
    FILE = open(DATA_PATH + savefile,"w")
    write_array(FILE,writedata,',')
    FILE.close()

def analyze_data(ref_files,transp):
    """ Plots histograms of the data.
    """
    data = np.zeros((0, 3))
    data_ind = []
    for n in range(len(ref_files)):
        target_file = DATA_PATH + ref_files[n][:-4] + "REF.txt"
        tmp = read_array(target_file,',') + transp[n]
        data = np.r_[data, tmp]
        data_ind.append(tmp)
    
    pitch = data[:,1]
    pitch = pitch[ np.where(pitch > 10) ]
    unique = list(set(pitch))
    unique.sort()
    print "pitch classes:",unique
    print "nr of classes:",len(unique)+1
    
    xmin = unique[0]
    xmax = unique[-1]
    xbins = xmax-xmin
    
    # plot global statistics
    pylab.figure()
    pylab.subplot(211)
    pylab.hist(pitch,bins=xbins)
    pylab.xlim(xmin,xmax)
    pylab.title("global pitch classes histogram")
    pylab.subplot(212)
    pylab.hist(data[:,2],bins=13)
    pylab.xlim(-1,11)
    pylab.title("global chromas histogram")
    
    # plot individual pieces
    pylab.figure()
    nr = len(data_ind)
    for n in range(nr):
        pitch = data_ind[n][:,1]
        pitch = pitch[ np.where(pitch > 10) ]
        unique = list(set(pitch))
        unique.sort()
        print n,":",unique
#        xmin = unique[0]
#        xmax = unique[-1]
        # plot it
        pylab.subplot(nr,1,n+1)
        pylab.hist(pitch,bins=(xmax-xmin))
        pylab.xlim(xmin,xmax)
        pylab.title(ref_files[n])
    
    

def extract_from_directory():
    """ Extracts the features of all files in DATA_PATH
    """
    files = glob.glob(DATA_PATH+"*.wav")
    for file in files:
        audiofile = os.path.basename(file)
        print "################################################################"
        print "# PROCESSING:",audiofile
        print "################################################################"
        normalized_stft( audiofile )
        print "\nCalculating Target ..."
        transform_target_data( audiofile )
        print "\n\n"
    
    print "################################################################"
    print "# COMPLETED !!!"
    print "################################################################"

#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
    
    # calculate one file
    #normalized_stft("jazz2.wav")
#    normalized_stft("train13.wav")
    transform_target_data_old("train12.wav",1)
#    transform_target_data("Help.wav",0)
    
#    convert_to_midi("../tmp/train13REF.txt","../tmp/M05train13REF.txt")
    
    # analysis
#    analyze_data(["WeAreTheChampions.wav", "ATasteOfHoney.wav",
#                  "ICantGetNoSatisfaction.wav", "AgainKravitz.wav"],
#    analyze_data(["ATasteOfHoney.wav", "AgainKravitz.wav"],
#                  [0,0])
#    analyze_data(["Help.wav", "WithALittleHelpFromMyFriends.wav",
#                  "BeautifulStranger.wav", "M05train12.wav"],
#    analyze_data(["WithALittleHelpFromMyFriends.wav","M05train12.wav"],
#                  [0,0])
#    pylab.show()
    
    # go through all the examples
#    extract_from_directory()
