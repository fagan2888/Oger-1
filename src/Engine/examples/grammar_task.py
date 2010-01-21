# Trains an ESN on sentences created by a probabilistic context free grammar.
# The performance is evaluated by measuring the cosine between the network
# outputs and the true probabilities distributions over the words according to
# the grammar itself.

import grammars as g
import gzip
from Engine import reservoir_nodes

import mdp
import pylab as p
from numpy import *

from Engine.linear_nodes import RidgeRegressionNode

def cosine(x, y):
    return dot(x, y) / (linalg.norm(x) * linalg.norm(y))

if __name__ == "__main__":

    l = g.simple_pcfg()

    testdata = []

    # Test sentences all have structure 'N V the N that N V .'
    print 'Generating data...'

    for N1 in l.nouns:
        for V1 in l.verbs:
            for N2 in l.nouns:
                for N3 in l.nouns:
                    for V2 in l.verbs:
                        s = [N1, V1, 'the', N2, 'that', N3, V2, '.']
                        testdata.append(s)

    # Number of sentences to train on.
    N = 5000

    trainsents = [l.S() for i in range(N)]
    trainwords = []

    for i in trainsents:
        trainwords.extend(i)

    testwords = []
    for i in testdata:
        testwords.extend(i)

    Nx = len(trainwords)

    vocab = l.nouns + l.verbs + [l.THAT] + [l.WITH] + [l.FROM] + [l.THE]\
            + [l.END]
    inputs = len(vocab)
    translate = dict([(vocab[i], i) for i in range(inputs)])

    reservoir = reservoir_nodes.ReservoirNode(inputs, 100, input_scaling = 1)
    readout = RidgeRegressionNode()

    # Build MDP network.
    flow = mdp.Flow([reservoir, readout], verbose=1)

    flownode = mdp.hinet.FlowNode(flow)

    # Contstruct a suitable train data matrix 'x'.
    indices = [translate[i] for i in trainwords]
    
    x = zeros((Nx, inputs))
    x[arange(Nx), array(indices)] = 1

    # y contains a timeshifted version of the data in x.
    y = mat(zeros((Nx, inputs)))
    y[:-1, :] = x[1:, :]
    y[-1, :] = x[0, :]

    # Open file with true probabilities.
    fin = gzip.open('trueprobs_simple_pcfg.txt.gz')
    dat = fin.readlines()
    fin.close()

    testprobs = []
    for i in dat:
        line = i.strip().split()
        datapoint = array([float(j) for j in line])
        testprobs.append(datapoint)

    print "Training..."
    flownode.train(x, y)
    flownode.stop_training()

    print "Testing..."

    # Contstruct a suitable train data matrix 'x' for the testset.
    indices = [translate[i] for i in testwords]
    
    Nx = len(testwords)
    x = zeros((Nx, inputs))
    x[arange(Nx), array(indices)] = 1

    # Save test results in ytest.
    ytest = flownode(x)

    results = [cosine(ytest[i], testprobs[i + 1]) for i in range(Nx - 1)]
    print 'Average cosine between outputs and ground distributions:', mean(results)
    results = array(results)[:-7]
    results = results.reshape(((Nx - 8) / 8, 8))
    means = mean(results, 0)
    example = ['boys', 'see', 'the', 'clowns', 'that', 'pigs', 'follow', '<end>']
    p.xticks(arange(8), example)
    p.bar(arange(8), means)
    p.show()
    print 'Finished'

