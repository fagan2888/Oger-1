#===============================================================================
# General simulation procedure for a Classification Experiment
#===============================================================================

import numpy as np
import random
import shelve


def train_model(trainin, trainout, model):
    """ Trains an MDP model.
    """
    model.train_looprange = range(len(trainin))
    if model.randrange:
        random.shuffle(model.train_looprange)
    
    # make the training for MDP ESN
    if model.type == 'ESN_mdp':
        # set train noise
        model._flow[1].nodes[0].reservoir.setNoise(model.trainnoise)
        
        # train the model
        for n in model.train_looprange:
            # reset state
            if model.reset_state:
                model._flow[1].nodes[0].reservoir.resetState()
            
            # output activation function
            if model.output_act == "tanh":
                targ = np.arctanh(trainout[n])
            else:
                targ = trainout[n]
                
            model.train(trainin[n], targ)
        
        model.stop_training()
        
        if model.print_W==1:
            print "output weights:"
            print "\tmean: ", model._flow[2].W.mean(), "\tmax: ", \
                              abs(model._flow[2].W).max()

    # make the training for aureservoir ESN
    if model.type == 'ESN_au':
        # set train noise
        model.setNoise(model.trainnoise)
        
        # construct training data
        inp = np.zeros((trainin[0].shape[1],0))
        outp = np.zeros((trainout[0].shape[1],0))
        for n in model.train_looprange:
            inp = np.c_[inp, trainin[n].T]
            outp = np.c_[outp, trainout[n].T]
                
        # train the model
        if model.ds==1:
            model.train(inp.copy(),outp.copy(),model.maxdelay+1)
        else:
            if hasattr(model, "washout"):
                washout = model.washout
            else:
                washout = 0
            model.train(inp.copy(),outp.copy(),washout)
        
        # print Wouts
        if model.ds != 2:
            print "output weights:"
            print "\tmean: ", model.getWout().mean(), "\tmax: ", \
                  abs(model.getWout()).max()
        
        # print delay
        if model.ds == 1:
            model.delays = np.zeros((model.getOutputs(),model.getSize()+model.getInputs()))
            model.getDelays(model.delays)
            print "trained delays:"
            print model.delays
            print "\tmean: ", model.delays.mean(), "\tmax: ", model.delays.max()

    # make the training for MDP ESN
    if model.type == 'ESN_statecompr':
        # set train noise
        model.setNoise(model.trainnoise)
        
        # train the model
        for n in model.train_looprange:
            # reset state
            if model.reset_state:
                model.resetState()
            
            # output activation function
            if model.output_act == "tanh":
                targ = np.arctanh(trainout[n][-1])
            else:
                targ = trainout[n][-1]
                
            model.train(trainin[n], targ.reshape(1,-1))
        
        model.stop_training()
        
#        if model.print_W==1:
#            print "output weights:"
#            print "\tmean: ", model._flow[2].W.mean(), "\tmax: ", \
#                              abs(model._flow[2].W).max()

    # SVM training with mean inputs
    if model.type == 'SVM_mean':
        for n in model.train_looprange:
            # calc mean of features
            inp = trainin[n].mean(0).reshape(1,-1)
            outp = trainout[n][0].reshape(1,-1)
            model.train(inp, outp)
        model.stop_training()
    
    # SVM training with anytime inputs
    if model.type == 'SVM_any':
        for n in model.train_looprange:
            model.train(trainin[n], trainout[n])
        model.stop_training()

def test_model(testin, model):
    """ Simulates an MDP model.
    """
    model.test_looprange = range(len(testin))
    if model.randrange:
        random.shuffle(model.test_looprange)

    testout = []
    
    # calc the algorithm for ESN
    if model.type == 'ESN_mdp':
        # set test noise
        model._flow[1].nodes[0].reservoir.setNoise(model.testnoise)
        
        # simulate the model
        for n in model.test_looprange:
            # reset state
            if model.reset_state:
                model._flow[1].nodes[0].reservoir.resetState()
            
            tmp = model(testin[n])
            
            # output activation function
            if model.output_act == "tanh":
                tmp = np.tanh(tmp)
            
            testout.append(tmp)

    # calc the algorithm for aureservoir ESN
    if model.type == 'ESN_au':
        # set test noise
        model.setNoise(model.testnoise)
        
        # get nr of outputs
        if model.ds != 2:
            outputs = model.getOutputs()
        else:
            outputs = model.getNetwork(0).getOutputs()
        
        # simulate the model
        for n in model.test_looprange:
            outp = np.zeros((outputs,testin[n].shape[0]))
            model.simulate(testin[n].T.copy(), outp)
            testout.append(outp.T)

    # calc the algorithm for ESN with state compression
    if model.type == 'ESN_statecompr':
        # set test noise
        model.setNoise(model.testnoise)
        
        # simulate the model
        for n in model.test_looprange:
            # reset state
            if model.reset_state:
                model.resetState()
            
            tmp = model(testin[n])
            
            # output activation function
            if model.output_act == "tanh":
                tmp = np.tanh(tmp)
            
            testout.append(tmp)

    # SVM simulation with mean inputs
    if model.type == 'SVM_mean':
        for n in model.test_looprange:
            # calc mean of features
            inp = testin[n].mean(0).reshape(1,-1)
            outp = model(inp)
            testout.append(outp)
            
    # SVM simulation with anytime inputs
    if model.type == 'SVM_any':
        for n in model.test_looprange:
            tmp = model(testin[n])
            testout.append(tmp)
    
    return testout


#===============================================================================
# high level interface
#===============================================================================

def run_simulation(model_func,mf_args,data_func,df_args,datafile,analyse_data=0):
    """ Runs a classification experiment (in one single fold).
    
    model_func    --   function to create the model
    mf_args       --   a list with arguments for model_func
    data_func     --   function to create the dataset
    df_args       --   a list with arguments for data_func
    datafile      --   where to store the results
    analyse_data  --   plots information about the training data
    """
    model = model_func(*mf_args)
    
    print "Load Training Data ..."
    trainin,trainout,testin,target = data_func(*df_args)
    
    # training data analysis
    if analyse_data == 1:
        import pylab
        pylab.figure()
        pylab.plot(trainin[0])
        pylab.figure()
        pylab.plot(testin[0])
        pylab.show()
        exit(0)
    
    print "Training ..."
    train_model(trainin, trainout, model)
    
    # remove unneeded training data to save RAM
    del trainin
    del trainout

    print "Testing ..."
    testout = test_model(testin, model)
    
    print "Writing results to disk ..."
    data = shelve.open(datafile)
    data["testout"] = testout
    data["target"] = target
    data["test_looprange"] = model.test_looprange
    data["datatype"] = model.dtype
    data.close()
    
    print "... finished !"

def run_simulation_multifold(model_func,mf_args,data_func,df_args,datafile,folds=3):
    """ Runs a classification experiment with n-fold crossvalidation.
    
    model_func    --   function to create the model
    mf_args       --   a list with arguments for model_func
    data_func     --   function to create the dataset
    df_args       --   a list with arguments for data_func
    datafile      --   where to store the results
    folds            --   number of folds
    """
    
    print "\n----------------------------------------------"
    print folds,"-fold Crossvalidation"
    print "----------------------------------------------"
    
    testout_all = []
    target_all = []
    looprange_all = []
    
    for curfold in range(folds):
        print "\n------- FOLD",curfold,"-------"
        
        model = model_func(*mf_args)
    
        print "Load Training Data ..."
        trainin,trainout,testin,target = data_func(*df_args)
        
        print "Training ..."
        train_model(trainin, trainout, model)
        
        # remove unneeded training data to save RAM
        del trainin
        del trainout
    
        print "Testing ..."
        testout = test_model(testin, model)
        
        # append data to list
        testout_all += testout
        target_all += target
        
        # add to looprange the previous length
        prevle = len(looprange_all)
        for it in model.test_looprange:
            looprange_all.append(it+prevle)

    print "\n----------------------------------------------"
    print "Writing results to ",datafile,"..."
    data = shelve.open(datafile)
    data["testout"] = testout_all
    data["target"] = target_all
    data["test_looprange"] = looprange_all
    data["datatype"] = model.dtype
    data.close()
    
    print "... finished !"
