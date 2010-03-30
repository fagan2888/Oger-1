import Engine
import pylab
import mdp
import time

if __name__ == "__main__":

    inputs = 1
    timesteps = 10000
    washout = 30

    nx = 4
    ny = 1

    [x, y] = Engine.datasets.narma30(sample_len=1000)

    # construct individual nodes
    reservoir = Engine.nodes.ReservoirNode(inputs, 100, input_scaling=0.05)
    readout = Engine.nodes.RidgeRegressionNode()

    # build network with MDP framework
    flow = mdp.parallel.ParallelFlow([reservoir, readout], verbose=1)
    
    scheduler = mdp.parallel.ThreadScheduler(n_threads=2, verbose=True)
#    scheduler = mdp.parallel.ProcessScheduler(n_processes=2, verbose=True)
#    scheduler = mdp.parallel.pp_support.LocalPPScheduler(ncpus=2, max_queue_length=0, verbose=True)

    data = [x[0:-1], zip(x[0:-1], y[0:-1])]
    
    # train the flow 
    flow.train(data, scheduler)
    
    scheduler.shutdown()

    #apply the trained flow to the training data and test data
    trainout = flow(x[0])
    testout = flow(x[9])

    print "NRMSE: " + str(Engine.utils.nrmse(y[9], testout))

    #plot the input
    pylab.subplot(nx, ny, 1)
    pylab.plot(x[0])
    
    #plot everything
    pylab.subplot(nx, ny, 2)
    pylab.plot(trainout, 'r')
    pylab.plot(y[0], 'b')

    pylab.subplot(nx, ny, 3)
    pylab.plot(testout, 'r')
    pylab.plot(y[9], 'b')
    
    pylab.subplot(nx, ny, 4)
    pylab.plot(flow.inspect(reservoir))
    pylab.show()

