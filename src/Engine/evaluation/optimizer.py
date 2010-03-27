'''
Created on Feb 9, 2010

@author: dvrstrae
'''
import Engine
import mdp
import pylab
import itertools

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
        self.errors = []

        # Construct the parameter space
        # Loop over all nodes that need their parameters set
        for node_key in self.optimization_dict.keys():
            # Loop over all parameters that need to be set for that node
            for parameter in self.optimization_dict[node_key].keys():
                # Append the parameter name and ranges to the corresponding lists
                self.parameter_ranges.append((self.optimization_dict[node_key])[parameter])
                self.paramspace_dimensions.append(len(self.parameter_ranges[-1]))
                self.parameters.append((node_key, parameter))

        # Construct all combinations
        self.param_space = list(itertools.product(*self.parameter_ranges))
        

    def grid_search (self, data, flow, cross_validate_function, *args, **kwargs):
        ''' Do a combinatorial grid-search of the given parameters and given parameter ranges, and do cross-validation of the flowNode
            for each value in the parameter space.
            Input arguments are:
            - data: a list of iterators which would be passed to MDP flows
            - flow : the MDP flow to do the grid-search on
        '''
        self.errors = mdp.numx.zeros(self.paramspace_dimensions)
        node_set = set()
        # Loop over all points in the parameter space
        for paramspace_index_flat, parameter_values in enumerate(self.param_space):
            # Set all parameters of all nodes to the correct values
            for parameter_index, node_parameter in enumerate(self.parameters):
                # Add the current node to the set of nodes whose parameters are changed, and which should be re-initialized 
                node_set.add(node_parameter[0])
                node_parameter[0].__setattr__(node_parameter[1], parameter_values[parameter_index])
            
            # Re-initialize all nodes that have the initialize method (e.g. reservoirs nodes)
            for node in node_set:
                if hasattr(node, 'initialize'):
                    node.initialize()
           
            # After all node parameters have been set and initialized, do the cross-validation
            paramspace_index_full = mdp.numx.unravel_index(paramspace_index_flat, self.paramspace_dimensions) 
            self.errors[paramspace_index_full] = mdp.numx.mean(Engine.evaluation.validate(data, flow, self.loss_function, cross_validate_function, *args, **kwargs))

    def plot_results(self, node_param_list=None):
        ''' Plot the results of the optimization. 
        
            Works for 1D and 2D scans, yielding a 2D resp. 3D plot of the parameter(s) vs. the error.
            node_param_list is an optional argument. It is a list of (node, param_string) tuples.
            Before plotting, the mean will be taken over all these node.param_string combinations,
            which is useful to plot/reduce multi-dimensional parameter sweeps.
        '''

        errors_to_plot, var_errors, parameters = self.mean_and_var(node_param_list)
             
        # If we have ranged over only one parameter
        if errors_to_plot.ndim == 1:
            # Average errors over folds
            pylab.ion()
            print "Warning: parameter_ranges still wrong!"
            pylab.errorbar(self.parameter_ranges[0], errors_to_plot, var_errors)
            pylab.xlabel(str(parameters[0][0]) + '.' + parameters[0][1])
            pylab.ylabel(self.loss_function.__name__)
            pylab.show()
        elif errors_to_plot.ndim == 2:
            pylab.ion()
            pylab.imshow(self.errors, interpolation='nearest')
            pylab.ylabel(str(parameters[0][0]) + '.' + parameters[0][1])
            pylab.xlabel(str(parameters[1][0]) + '.' + parameters[1][1])

            pylab.colorbar()
            pylab.show()
        else:
            raise Exception("Too many parameter dimensions to plot: " + str(errors_to_plot.ndim))

    def mean_and_var(self, node_param_list):
        ''' Return a tuple containing the mean and variance of the errors over a certain parameter.
            
            Gives the mean/variance of the errors w.r.t. the parameter given by 
            node_param_list, where each element is a (node, parameter) tuple.
            If the list has only one element, the variance w.r.t this parameter
            is also returned, otherwise the second return argument is None.
        '''
        # In case of an empty list, we just return the errors
        if node_param_list is None:
            return self.errors, None, self.parameters
        
        # Check if we have the requested node.parameter combinations in the optimization_dict
        for node, param_string in node_param_list:    
            if not node in self.optimization_dict:
                raise Exception('Cannot take the mean, given node ' + str(node) + ' is not in optimization_dict.')
            if not param_string in self.optimization_dict[node]:
                raise Exception("Cannot take the mean, given parameter '" + param_string + "' is not in optimization_dict.")

        # take a copy, so we can eliminate some of the parameters later
        # However, we don't want to do a deep copy, just create a new list 
        # with references to the old elements, hence the [:]
        parameters = self.parameters[:]
        errors = self.errors[:]
        
        # Loop over all parameters in node_param_list and iteratively compute 
        # the mean
        for node, param_string in node_param_list:
            # Find the axis we need to take the mean across
            axis = parameters.index((node, param_string))
            
            # Finally, return the mean and variance
            mean_errors = mdp.numx.mean(errors, axis)
            
            # In case we take the mean over only one dimension, we can return
            # the variance as well
            if len(node_param_list) == 1:
                # Use ddof = 1 to mimic matlab var
                var = mdp.numx.var(errors, axis, ddof=1)
            else:
                var = None
            
            # Remove the corresponding dimension from errors and parameters for
            # the next iteration of the for loop
            errors = mean_errors
            parameters.remove((node, param_string))
            
        return errors, var, parameters
         

    def get_minimal_error(self, node_param_list=None):
        '''Return the minimal error, and the corresponding parameter values as a tuple:
        (error, param_values), where param_values is a dictionary of dictionaries,  
        with the key of the outer dictionary being the node, and inner dictionary
        consisting of (parameter:optimal_value) pairs.
        If the optional argument node_param_list is given, first the mean of the
        error will be taken over all (node, parameter) tuples in the node_param_list
        before taking the minimum
        '''
        if self.errors is None:
            raise Exception('Errors array is empty. No optimization has been performed yet.')

        errors, _, parameters = self.mean_and_var(node_param_list)
        min_parameter_dict = {}
        minimal_error = mdp.numx.amin(errors)
        min_parameter_indices = mdp.numx.unravel_index(mdp.numx.argmin(errors), errors.shape)

        for index, param_d in enumerate(parameters):
            global_param_index = self.parameters.index(param_d)
            opt_parameter_value = self.parameter_ranges[global_param_index][min_parameter_indices[index]]
            # If there already is an entry for the current node in the dict, add the 
            # optimal parameter/value entry to the existing dict
            if param_d[0] in min_parameter_dict:
                min_parameter_dict[param_d[0]][param_d[1]] = opt_parameter_value
            # Otherwise, create a new dict
            else:
                min_parameter_dict[param_d[0]] = {param_d[1] : opt_parameter_value}

        return (minimal_error, min_parameter_dict) 
