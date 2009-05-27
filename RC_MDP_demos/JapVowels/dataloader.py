#===============================================================================
# Reads and preprocesses the Japanese Vowel Dataset
# (after Jaegers matlab script)
#===============================================================================

from scipy import io
import numpy as np
import shelve


def load_data():
    """ Reads the Japanese Vowel Dataset from textfiles as in Jaegers paper
    and puts it into python lists / numpy arrays.
    """
    aetrain = io.read_array("./JaegerPaper/data/ae.train")
    aetest = io.read_array("./JaegerPaper/data/ae.test")
    
    # value for bias input
    bias = 0.07
    
    # aetrain and aetest contain the 12-dim time series, which have
    # different lengthes, concatenated vertically and separated by ones(1,12)
    # rows. We now sort them into lists, such that each element represents
    # one time series
    trainInputs = []
    readindex = 0
    for c in range(270):
        l = 0
        while aetrain[readindex,0] != 1.0:
            l += 1
            readindex += 1
        readindex += 1
        trainInputs.append( aetrain[readindex-l-1:readindex-1,:] )
        # add bias input and input which indicates the length
        trainInputs[c] = np.c_[ trainInputs[c], bias*np.ones((l,1)), 
                                (l/30.)*np.ones((l,1)) ]
    
    # now the same with the test inputs
    testInputs = []
    readindex = 0
    for c in range(370):
        l = 0
        while aetest[readindex,0] != 1.0:
            l += 1
            readindex += 1
        readindex += 1
        testInputs.append( aetest[readindex-l-1:readindex-1,:] )
        # add bias input and input which indicates the length
        testInputs[c] = np.c_[ testInputs[c], bias*np.ones((l,1)), 
                               (l/30.)*np.ones((l,1)) ]

    # produce teacher signals. For each input time series of size N x 12 this
    # is a time series of size N x 9, all zeros except in the column indicating
    # the speaker, where it is 1.
    trainOutputs = []
    for c in range(270):
        l = trainInputs[c].shape[0]
        teacher = np.zeros((l,9))
        speakerIndex = int( np.ceil( (c+1)/30. ) - 1 )
        teacher[:,speakerIndex] = np.ones(l)
        trainOutputs.append( teacher )
    
    # produce test output signal
    testOutputs = []
    speakerIndex = 0
    blockCounter = 0
    blockLengthes = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    for c in range(370):
        if blockCounter == blockLengthes[speakerIndex]:
            speakerIndex += 1
            blockCounter = 0
        blockCounter += 1
        l = testInputs[c].shape[0]
        teacher = np.zeros((l,9))
        teacher[:,speakerIndex] = np.ones(l)
        testOutputs.append( teacher )
        
    return trainInputs, trainOutputs, testInputs, testOutputs


def calc_mins(data):
    """ Calculates the minimum of each channel.
    """
    channels = data[0].shape[1]
    
    # find minimum of each channel
    mins = np.ones(channels) * 10
    for vowel in data:
        for n in range(channels):
            min = vowel[:,n].min()
            if min < mins[n]:
                mins[n] = min
    return mins

def shift_data(data, shifts):
    """ Shifts every channel n of the data by shift[n].
    """
    channels = data[0].shape[1]
    
    for vowel in data:
        for n in range(channels):
            vowel[:,n] += shifts[n]
    return data

def normalize_data_min0(data):
    """ Normalizes each channel, so that the minimum is 0.
    """
    mins = calc_mins(data)
    mins[12] = 0. # don't shift constant channel
    shifts = (-1) * mins
    
    shdata = shift_data(data, shifts)
    return shdata, shifts

def save_data(filename="voweldata.dat"):
    """ Saves the shifted Vowel Data to disk.
    """
    print "generating data ..."
    voweldata = load_data() 
    
    # shift training data so that minimum is 0
    trainInShifted, shifts = normalize_data_min0(voweldata[0])
    # shift also test data by the same amount
    testInShifted = shift_data(voweldata[2], shifts)
    
    print "writing data to disk ..."
    data = shelve.open(filename)
    data["trainInputs"] = trainInShifted
    data["trainOutputs"] = voweldata[1]
    data["testInputs"] = testInShifted
    data["testOutputs"] = voweldata[3]
    data["shifts"] = shifts
    data.close()
    print "... finished !"


#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
#    load_data()
    save_data()

