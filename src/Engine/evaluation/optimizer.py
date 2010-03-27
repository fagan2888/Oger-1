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
                self.parameters.append({'node':node_key, 'parameter':parameter})

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
        # Loop over all points in the parameter space
        for paramspace_index_flat, parameter_values in  mdp.utils.progressinfo(enumerate(self.param_space), style='timer', length=len(self.param_space)):
            # Set all parameters of all nodes to the correct values
            node_set = set()
            for parameter_index, node_parameter in enumerate(self.parameters):
                # Add the current node to the set of nodes whose parameters are changed, and which should be re-initialized 
                node_set.add(node_parameter['node'])
                node_parameter['node'].__setattr__(node_parameter['parameter'], parameter_values[parameter_index])
            
            # Re-initialize all nodes that have the initialize method (e.g. reservoirs nodes)
            for node in node_set:
                if hasattr(node, 'initialize'):
                    node.initialize()
           
            # After all node parameters have been set and initialized, do the cross-validation
            paramspace_index_full = mdp.numx.unravel_index(paramspace_index_flat, self.paramspace_dimensions) 
            self.errors[paramspace_index_full] = mdp.numx.mean(Engine.evaluation.validate(data, flow, self.loss_function, cross_validate_function, progress=False, *args, **kwargs))

    def plot_results(self):
        ''' Plot the results of the optimization. Works for 1D and 2D scans, yielding a 2D resp. 3D plot of the parameter(s) vs. the error.
        '''

        # If we have ranged over only one parameter
        if len(self.parameters) == 1:
            # Average errors over folds
            mean_errors = map(mdp.numx.mean, self.errors)
            
            #Variance across folds
            var_errors = map(mdp.numx.var, self.errors)
            
            pylab.ion()
            pylab.errorbar(self.parameter_ranges[0], mean_errors, var_errors)
            pylab.xlabel(str(self.parameters[0]['node']) + '.' + self.parameters[0]['parameter'])
            pylab.ylabel(self.loss_function.__name__)
            pylab.show()
        elif len(self.parameters) == 2:
            pylab.ion()
            pylab.imshow(self.errors, interpolation='nearest')
            pylab.ylabel(str(self.parameters[0]['node']) + '.' + self.parameters[0]['parameter'])
            pylab.xlabel(str(self.parameters[1]['node']) + '.' + self.parameters[1]['parameter'])

            pylab.colorbar()
            pylab.show()
        else:
            print ("Plotting only works for 1D or 2D parameter sweeps!")

    def mean_across_parameter(self):
        pass

    def get_minimal_error(self):
        '''Return the minimal error, and the corresponding parameter values as a tuple:
        (error, param_values), where param_values is a dictionary of dictionaries,  
        with the key of the outer dictionary being the node, and inner dictionary
        consisting of (parameter:optimal_value) pairs.
        '''
        if self.errors is None:
            raise Exception('Errors array is empty. No optimization has been performed yet.')

        min_parameter_dict = {}
        minimal_error = mdp.numx.amin(self.errors)
        min_parameter_indices = mdp.numx.unravel_index(mdp.numx.argmin(self.errors), self.errors.shape)

        for index, param_d in enumerate(self.parameters):
            opt_parameter_value = self.parameter_ranges[index][min_parameter_indices[index]]
            # If there already is an entry for the current node in the dict, add the 
            # optimal parameter/value entry to the existing dict
            if param_d['node'] in min_parameter_dict:
                min_parameter_dict[param_d['node']][param_d['parameter']] = opt_parameter_value
            # Otherwise, create a new dict
            else:
                min_parameter_dict[param_d['node']] = {param_d['parameter'] : opt_parameter_value}

        return (minimal_error, min_parameter_dict) 
    
    
    

    
class ParameterSettingNode(mdp.Node):
    
    def __init__(self, flow, loss_function, cross_validation, input_dim=None, output_dim=None, dtype=None, *args, **kwargs):
        super(ParameterSettingNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.flow = flow
        self.loss_function = loss_function
        self.cross_validation = cross_validation
        
        # TODO: this is a bit messy...
        self.args = args
        self.kwargs = kwargs
        
    def execute(self, x):
        params = x[0]
        data = x[1]
        
        # Set all parameters of all nodes to the correct values
        node_set = set()
        for node_index, node_dict in params.items():
            for parameter, value in node_dict.items():
                node = self.flow[node_index]
                node_set.add(node)
                node.__setattr__(parameter, value)
        
        # Re-initialize all nodes that have the initialize method (e.g. reservoirs nodes)
        for node in node_set:
            if hasattr(node, 'initialize'):
                node.initialize()
       
        # TODO: could this set of functions also be a parameter?
        return [mdp.numx.mean(Engine.evaluation.validate(data, self.flow, self.loss_function, self.cross_validation, progress=False, *self.args, **self.kwargs))]
        
    def is_trainable(self):
        return False

@mdp.extension_method("parallel", Optimizer)
def grid_search (self, data, flow, cross_validate_function, *args, **kwargs):
    ''' Do a combinatorial grid-search of the given parameters and given parameter ranges, and do cross-validation of the flowNode
        for each value in the parameter space.
        Input arguments are:
        - data: a list of iterators which would be passed to MDP flows
        - flow : the MDP flow to do the grid-search on
    '''
    
    if not hasattr(self, 'scheduler') or self.scheduler is None:
        err = ("No scheduler was assigned to the Optimizer so cannot run in parallel mode.")
        raise Exception(err)
        
    self.errors = mdp.numx.zeros(self.paramspace_dimensions)
    
    data_parallel = []
    
    # Loop over all points in the parameter space
    for paramspace_index_flat, parameter_values in enumerate(self.param_space):
        params = {}

        for parameter_index, node_parameter in enumerate(self.parameters):
            node_index = flow.flow.index(node_parameter['node'])
            if not node_index in params:
                params[node_index] = {}
                
            params[node_index][node_parameter['parameter']] = parameter_values[parameter_index]
        
        data_parallel.append([[params, data]])

    parallel_flow = mdp.parallel.ParallelFlow([ParameterSettingNode(flow, self.loss_function, cross_validate_function, *args, **kwargs),])

    results = parallel_flow.execute(data_parallel, scheduler=self.scheduler)

    i = 0
    for paramspace_index_flat, parameter_values in enumerate(self.param_space):
        paramspace_index_full = mdp.numx.unravel_index(paramspace_index_flat, self.paramspace_dimensions) 
        self.errors[paramspace_index_full] = results[i]
        i+=1
        
    self.scheduler.shutdown()
