#===============================================================================
# example output-plots for the dafx paper
#===============================================================================

import shelve
import numpy as np
import pylab


# the files with the results
svm_results = "results/svm_test.dat"
esn_results = "results/esn_test.dat"

# the pitch classes (see console output during simulation
classes = [0.0, 53.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0,
           66.0, 67.0, 68.0, 69.0, 71.0, 73.0, 74.0, 76.0]


#------------------------------------------------------------------------------ 

# read result files
data = shelve.open(svm_results)
svm_testout = data["testout"]
svm_target = data["target"]
svm_looprange = data["test_looprange"]
data.close()
data = shelve.open(esn_results)
esn_testout = data["testout"]
esn_target = data["target"]
esn_looprange = data["test_looprange"]
data.close()

# make anytime outputs
examples = len(svm_looprange)
outputs = len(classes)
svm_target_any = np.zeros((0,outputs))
svm_testout_any = np.zeros((0,outputs))
esn_target_any = np.zeros((0,outputs))
esn_testout_any = np.zeros((0,outputs))
for n in range(examples):
    svm_testout_any = np.r_[svm_testout_any, svm_testout[n]]
    svm_target_any = np.r_[svm_target_any, svm_target[svm_looprange[n]]]
    esn_testout_any = np.r_[esn_testout_any, esn_testout[n]]
    esn_target_any = np.r_[esn_target_any, esn_target[esn_looprange[n]]]

# calc max index
svm_ind = svm_testout_any.argmax(1)
esn_ind = esn_testout_any.argmax(1)
target_ind = svm_target_any.argmax(1)

le = svm_testout_any.shape[0]
svm_midi = np.zeros(le)
esn_midi = np.zeros(le)
target_midi = np.zeros(le)
for n in range(le):
    svm_midi[n] = classes[svm_ind[n]]
    esn_midi[n] = classes[esn_ind[n]]
    target_midi[n] = classes[target_ind[n]]

#print classes
#print svm_ind[1000:1050]
#print svm_midi[1000:1050]
#print esn_ind[1000:1050]
#print esn_midi[1000:1050]
#print target_ind[1000:1050]
#print target_midi[1000:1050]

import matplotlib.mlab as mlab
#from pylab import figure, show


# make plots
pylab.figure()

xmin = 300
xmax = 550
ymin = 50
ymax = 70

pylab.subplot(121)
pylab.fill(np.arange(le), target_midi, edgecolor='0.8', facecolor='0.8')
pylab.plot(esn_midi, '.', color='0.0')
pylab.xlim(xmin,xmax)
pylab.ylim(ymin,ymax)
pylab.title('Reservoir computing network')
#pylab.ylabel('MIDI Note')
#pylab.xlabel('Frame')
pylab.grid(True)

pylab.subplot(122)
pylab.fill(np.arange(le), target_midi, edgecolor='0.8', facecolor='0.8')
pylab.plot(svm_midi, '.', color='0.0')
pylab.xlim(xmin,xmax)
pylab.ylim(ymin,ymax)
pylab.title('Support vector machine')
#pylab.ylabel('MIDI Note')
#pylab.xlabel('Frame')
pylab.grid(True)

pylab.show()




#fig = pylab.figure()
#ax = fig.add_subplot(111)
#ax.plot(np.arange(le), target_midi,'.')
#ax.fill_between(np.arange(le), 0, target_midi)

#pylab.plot(esn_midi)
#pylab.plot(svm_midi)
#pylab.show()


