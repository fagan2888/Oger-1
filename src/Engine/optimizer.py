'''
Created on Feb 9, 2010

@author: dvrstrae
'''
import mdp
import pylab
import itertools
from Engine import crossvalidation 

class Optimizer(object):
    def __init__(self, optimization_dict, loss_function, **kwargs):
        ''' Construct an Optimizer object.
            optimization_dict: a dictionary of dictionaries. 
                In the top-level dictionary, every key is a node, and the corresponding item is again a dictionary 
                where the key is the parameter name to be optimize and the corresponding item is a list of values for the parameter
                Example: to gridsearch the spectral radius and input scaling of reservoirnode1, and the input scaling of reservoirnode2 over the range .1:.1:1 this would be:
                opt_dict = {reservoirnode1: {'spec_radius', np.arange(.1,1,.1)}, 'input_scaling', np.arange(.1,1,.1)}}, reservoirnode2: {'input_scaling', np.arange(.1,1,.1)}}
            loss_function: the function used to compute the loss
        '''
        # Initialize attributes
        self.optimization_dict = optimization_dict 
        self.loss_function = loss_function
        self.parameter_ranges = []
        self.paramspace_dimensions = []
        self.parameters = []

        # Construct the parameter space
        # Loop over all nodes that need their parameters set
        for node_key in self.optimization_dict.keys():
            # Loop over all parameters that need to be set for that node
            for parameter in self.optimization_dict[node_key].keys():
                # Append the parameter name and ranges to the corresponding lists
                self.parameter_ranges.append((self.optimization_dict[node_key])[parameter])
                self.paramspace_dimensions.append(len(self.parameter_ranges[-1]))
                self.parameters.append({'node':node_key,'parameter':parameter})

        # Construct all combinations
        self.param_space = list(itertools.product(*self.parameter_ranges))
        self.errors = mdp.numx.zeros(self.paramspace_dimensions)

    def grid_search (self,x,y, flowNode, n_folds=5, crossvalidation_function = crossvalidation.cross_val_random):
        ''' Do a combinatorial grid-search of the given parameters and given parameter ranges, and do cross-validation of the flowNode
            for each value in the parameter space.
            Input arguments are:
            - data: a list of iterators which would be passed to MDP flows
            - flowNode : the MDP flowNode to do the grid-search on
        '''
        # Loop over all points in the parameter space
        for paramspace_index_flat, parameter_values in enumerate(self.param_space):
            # Set all parameters of all nodes to the correct values
            for parameter_index, node_parameter in enumerate(self.parameters): 
                print str(node_parameter['node']) + '.' + str(node_parameter['parameter']) + ' = ' + str(parameter_values[parameter_index]), 
                node_parameter['node'].__setattr__(node_parameter['parameter'], parameter_values[parameter_index])
                # Reinitialize the node
                if hasattr(node_parameter['node'], 'initialize'):
                    node_parameter['node'].initialize()
            print
            # After all node parameters have been set and initialized, do the cross-validation
            paramspace_index_full = mdp.numx.unravel_index(paramspace_index_flat, self.paramspace_dimensions) 
            self.errors[paramspace_index_full] = mdp.numx.mean(crossvalidation.cross_validate(x,y,flowNode, self.loss_function, n_folds=n_folds, cross_validate_function = crossvalidation_function ))

    def plot_results(self):
        ''' Plot the results of the optimization. Works for 1D and 2D scans, yielding a 2D resp. 3D plot of the parameter(s) vs. the error.
        '''

        # If we have ranged over only one parameter
        if len(self.parameters)==1:
            # Average errors over folds
            mean_errors=map(mdp.numx.mean, self.errors)
            
            #Variance across folds
            var_errors = map(mdp.numx.var, self.errors)
            
            pylab.ion()
            pylab.errorbar(self.parameter_ranges[0], mean_errors, var_errors)
            pylab.xlabel(str(self.parameters[0]['node']) + '.' + self.parameters[0]['parameter'])
            pylab.ylabel(self.loss_function.__name__)
            pylab.show()
        elif len(self.parameters)==2:
            pylab.ion()
            pylab.imshow(self.errors)
            pylab.show()
        else:
            print ("Plotting only works for 1D or 2D parameter sweeps!")

    def mean_across_parameter(self):
        pass

    def global_optimum(self):
        pass
