#===============================================================================
# some example datasets of the everyday sounds database
#
# ATTENTION: old methods, not used ATM (only the variables)
#===============================================================================

import numpy as np
import fileinput

# path to the audio files
CLOSED_AUDIO_PATH = "/home/holzi/Datasets/closed/data/audio_library/audio_files/"

# path with example lists (classes)
CLOSED_LIST_PATH = "/home/holzi/Datasets/closed/data/general_everyday_sounds/lists/"

# all classes/examples
SOUND_LISTS = ["deformation.list", "explosion.list", "friction.list", "pour.list",
               "whoosh.list", "drip.list", "flow.list", "impact.list",
               "rolling.list", "wind.list"]


def read_filelist(listfile):
    """ Helper function to read from closed list files. 
    """
    filelist = []
    for line in fileinput.input(CLOSED_LIST_PATH+listfile):
        if len(line)>3: # to ignore newlines
            filelist.append(CLOSED_AUDIO_PATH+line[:-1]) # remove newline
    return filelist

def make_marsyas_collection_all(outfile="data/closed_all.mf"):
    """ Generates marsyas collections with labels out of all
    closed .list files (one vs. all classification).
    """
    FILE = open(outfile,"w")
    
    # go through all closed listfiles and write it in one marsyas playlist
    for file in SOUND_LISTS:
        filelist = read_filelist(file)
        label = file[:-5]
        for audiofile in filelist:
            # tab is the separator for labels
            FILE.write(audiofile+"\t"+label+"\n")
    
    print "Content written to ", outfile
    FILE.close()

def make_marsyas_collection_two(list1=SOUND_LISTS[0], list2=SOUND_LISTS[1],
                                path="data/"):
    """ Generates marsyas collections with labels for two
    closed .list files (one vs. one classification).
    """
    outfile = path+list1[:-5]+"-vs-"+list2[:-5]+".mf"
    FILE = open(outfile,"w")
    sound_lists = [list1, list2]
    
    for file in sound_lists:
        filelist = read_filelist(file)
        label = file[:-5]
        for audiofile in filelist:
            # tab is the separator for labels
            FILE.write(audiofile+"\t"+label+"\n")
    
    print "Content written to ", outfile
    FILE.close()

def get_two_class_dataset(nr=10, class1="wind.list", class2="rolling.list"):
    """ Returns training/testing data from two classes.
    
    class1   -- File with a list of soundfiles from class1.
    class2   -- File with a list of soundfiles from clss2.
    nr       -- Number of examples to (randomly) choose from both classes.
    
    returns  -- A tuble (signal,label) with the signals and the corresponding
                label for each sample.
    """
    # read all examples from class one in a list
    file = open(CLOSED_LIST_PATH+class1)
    sounds1 = []
    while 1:
        line = file.readline()
        if not line:
            break
        else:
            sounds1.append( line[:-1] )
    file.close()
    
    # read the second class
    file = open(CLOSED_LIST_PATH+class2)
    sounds2 = []
    while 1:
        line = file.readline()
        if not line:
            break
        else:
            sounds2.append( line[:-1] )
    file.close()
    
    # generate random class labels
    labels = np.random.randint(0,2,nr)
    
    # generate soundfile according to the class labels
    signal = np.array([])
    label = np.array([],dtype='int32')
    for n in range(nr):
        if labels[n] == 0:
            # first class
            index = np.random.randint(0,len(sounds1),1)[0]
            filename = sounds1[index]
            (tmp, sr, enc) = wavread( CLOSED_AUDIO_PATH + filename )
            tmplabels = np.zeros(len(tmp), dtype='int')
        else:
            # second class
            index = np.random.randint(0,len(sounds2),1)[0]
            filename = sounds2[index]
            (tmp, sr, enc) = wavread( CLOSED_AUDIO_PATH + filename )
            tmplabels = np.ones(len(tmp), dtype='int')
        signal = np.hstack((signal,tmp))
        label = np.hstack((label,tmplabels))
    
    return signal.reshape(-1,1), label.reshape(-1,1)


#===============================================================================
# main function
#===============================================================================

if __name__ == "__main__":
#    make_marsyas_collection_all()
    
    make_marsyas_collection_two(SOUND_LISTS[1],SOUND_LISTS[4])
    
#    import pylab
#    sig,lab = get_two_class_dataset(5)
#    print sig.shape, lab.shape
#    pylab.plot(sig)
#    pylab.plot(lab-0.5, 'r')
#    pylab.show()
