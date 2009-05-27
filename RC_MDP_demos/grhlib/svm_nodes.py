#===============================================================================
# libsvm wrappers for the MDP package
#===============================================================================

import mdp
import svm
import liblinear as ll
import numpy


class BinarySVMNode(mdp.Node):
    """ A binary support vector machine from libsvm.
    """
    
    def __init__(self, input_dim=None, output_dim=None, params=None, dtype='float64'):
        """ Initializes the SVM.
                
        params  --  class of type svm_parameter with all parameters
                    see libsvm documentation
        """
        super(BinarySVMNode, self).__init__(input_dim, output_dim, dtype)
        
        if not params:
            # make a linear SVM with C = 10
            params = svm.svm_parameter(kernel_type = svm.LINEAR, C = 10)
        self.parameters = params
        
        # variables for training
        self.X = numpy.zeros((0,self._input_dim), dtype=self._dtype)
        self.Y = numpy.zeros((0,self._output_dim), dtype=self._dtype)
        
        # list with models, parameters and labels for each output
        self.problems = []
        self.models = []
        self.labels = []

    def reset_model(self):
        """ Resets the model for new training.
        """
        self.X = numpy.zeros((0,self._input_dim), dtype=self._dtype)
        self.Y = numpy.zeros((0,self._output_dim), dtype=self._dtype)

    def is_invertible(self):
        return False
    
    def _train(self, x, labels):
        """ Collects the training data for learning.
        
        x       --   the input data
        labels  --   the corresponding labels for each datapoint in x
        """
        # append new data to training data
        self.X = numpy.r_[ self.X, x ]
        self.Y= numpy.r_[ self.Y, labels ]
    
    def _stop_training(self):
        """ Trains and creates the model.
        """
        # reset variables
        self.problems = []
        self.models = []
        self.labels = []
        self._dosim = numpy.zeros(self._output_dim, dtype='int')
        
        # finally generate the models
        for n in range(self._output_dim):
            # get labels (min,max)
            self.labels.append((self.Y[:,n].max(), self.Y[:,n].min()))
            if self.labels[n][0] == self.labels[n][1]:
                # apport simulation if there is only one label
                self._dosim[n] = 1
                self.problems.append(None)
                self.models.append(None)
                continue
            
            # construct problems
            self.problems.append( svm.svm_problem(self.Y[:,n], self.X) )
            
            # generate models
            self.models.append( svm.svm_model(self.problems[n], self.parameters) )
            
            # check if there are only 2 classes
            if self.models[n].get_nr_class() > 2:
                raise mdp.NodeException("Only binary classification possible with libsvm for now !")
            
        # reset data for training
        self.reset_model()

    def _execute(self, x):
        """ Executes simulation with input vector x.
        
        Returns the decision values for the two classes:
        a negative number (-1) corresponds to the first class,
        a positive number (1) to the second class.
        """
        steps = x.shape[0]
        y = numpy.zeros((steps,self._output_dim), dtype=self._dtype)
        
        # iterate through models and datasteps
        for i in range(self._output_dim):
            if self._dosim[i] == 1:
                y[:,i] = self.labels[i][0]
                continue
            
            for j in range(steps):
                # NOTE: this gives the desicion values, maybe not the best way ?
                y[j,i] = self.models[i].predict_values( x[j,:] )[self.labels[i]]
                
        return y


class BinaryLinearSVMNode(mdp.Node):
    """ A binary linear support vector machine from liblinear.
    """
    
    def __init__(self, input_dim=None, output_dim=None, dtype='float64',
                 C=1.0, solver_type=1, eps=0.01):
        """ Initializes the SVM.
          
        C            --  the cost of constraints violation
                         (default 1, we usually use 1 to 1000)
        solver_type  --  set type of solver (default 1):
                         0 -- L2-regularized logistic regression
                         1 -- L2-loss support vector machines (dual)
                         2 -- L2-loss support vector machines (primal)
                         3 -- L1-loss support vector machines (dual)
        eps          --  the stopping criterion (we usually use 0.01)
        """
        super(BinaryLinearSVMNode, self).__init__(input_dim, output_dim, dtype)
        
        # check solver type:
        if solver_type == 0:
            self.solver = ll.L2_LR
        elif solver_type == 1:
            self.solver = ll.L2LOSS_SVM_DUAL
        elif solver_type == 2:
            self.solver = ll.L2LOSS_SVM
        elif solver_type == 3:
            self.solver = ll.L1LOSS_SVM_DUAL
        else:
            raise mdp.NodeException("solver_type must be 0, 1, 2 or 3 !")
        
        # the other parameters
        self.C = C
        self.eps = eps
        
        # variables for training data
        self.X = numpy.zeros((0,self._input_dim), dtype=self._dtype)
        self.Y = numpy.zeros((0,self._output_dim), dtype=self._dtype)
        
        # list with models and labels for each output
        self.models = []
        self.labels = []
    
    def reset_model(self):
        """ Resets the model for new training.
        """
        self.X = numpy.zeros((0,self._input_dim), dtype=self._dtype)
        self.Y = numpy.zeros((0,self._output_dim), dtype=self._dtype)
    
    def is_invertible(self):
        return False
    
    def _train(self, x, labels):
        """ Collects the training data for learning.
        
        x       --   the input data
        labels  --   the corresponding labels for each datapoint in x
        """
        # append new data to training data
        self.X = numpy.r_[ self.X, x ]
        self.Y= numpy.r_[ self.Y, labels ]
    
    def _stop_training(self):
        """ Trains and creates the model.
        """
        # reset variables
        self.models = []
        self.labels = []
        self._dosim = numpy.zeros(self._output_dim, dtype='int')
        
        # go through all outputs individually
        for n in range(self._output_dim):
            # get indeces of the classes
            self.labels.append((self.Y[:,n].max(), self.Y[:,n].min()))
            
            # check if there is only one label, then abort simulation
            if self.labels[n][0] == self.labels[n][1]:
                self._dosim[n] = 1
                self.models.append(None)
                continue
            
            min_ind = numpy.where( self.Y[:,n] == self.labels[n][1] )[0]
            max_ind = numpy.where( self.Y[:,n] == self.labels[n][0] )[0]
            
            # check if there are more classes
            leng = min_ind.size + max_ind.size
            if(leng < self.X.shape[0]):
                raise mdp.NodeException("Only binary classification possible with" +\
                                        " liblinear - you defined multiple labels !")
            
            # bring class1 in liblinear format
            class1 = []
            for sample in self.X[max_ind]:
                class1.append(ll.vector2sparse(sample))
            
            # bring class2 in liblinear format
            class2 = []
            for sample in self.X[min_ind]:
                class2.append(ll.vector2sparse(sample))
            
            # now train the classifier
            data = [class1, class2]
            self.models.append( ll.LinearSVM.train(data, C=self.C,
                                                   mach=self.solver, eps=self.eps) )
        
        # reset training data
        self.reset_model()

    def _execute(self, x):
        """ Executes simulation with input vector x.
        
        Returns the decision values for the two classes:
        a negative number (-1) corresponds to the first class,
        a positive number (1) to the second class.
        """
        steps = x.shape[0]
        y = numpy.zeros((steps,self._output_dim), dtype=self._dtype)
        
        # construct data in liblinear format
        data = []
        for sample in x:
            data.append( ll.vector2sparse(sample) )
        
        # go through all outputs individually
        for n in range(self._output_dim):
            # check if we need to simulat this class
            if self._dosim[n] == 1:
                y[:,n] = self.labels[n][0]
                continue
            
            # make the simulation
            for m in range(steps):
                y[m,n] = self.models[n].predict_values( data[m] )[0]
        
        return y
