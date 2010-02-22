'''
Created on Feb 9, 2010

@author: dvrstrae
'''
from Engine.crossvalidation import cross_validate
from itertools import product

def grid_search (data,optimization_parameters, flowNode, error_function):
    ''' Do a combinatorial grid-search of the given parameters and given parameter ranges, and do cross-validation of the flowNode
        for each value in the parameter space.
        Input arguments are:
        - data: a list of iterators which would be passed to MDP flows
        - optimization_parameters: a dictionary of dictionaries. 
            In the top-level dictionary, every key is a node, and the corresponding item is again a dictionary 
            where the key is the parameter name to be optimize and the corresponding item is a list of values for the parameter
        - flowNode : the MDP flowNode to do the grid-search on
    '''
    errors=[]
    parameter_ranges = []
    # Loop over all nodes that need their parameters set
    for node_key in optimization_parameters.iterkeys():
        for parameter_range in optimization_parameters[node_key].values():
            parameter_ranges.append(parameter_range)
        print parameter_ranges
        # Construct all combinations
        param_space = product(*parameter_ranges)
        # Loop over all points in the parameter space
    for parameter_values in param_space:
        for node_key in optimization_parameters.iterkeys():
            # Set all individual parameters
            for parameter_index, parameter in enumerate(optimization_parameters[node_key].keys()):
                print node_key + '.' + parameter + ' = ' + str(parameter_values[parameter_index]), 
                #node_key.__setattr__(parameter, parameter_values[parameter_index])
        print
            # Reinitialize the node
            # node_key.initialize()
            # After all node parameters have been set and initialized, do the cross-validation
            # errors.append(cross_validate(data,flowNode, error_function, 5))'''
    #for i,j,k in param_space:
    #    print i,j,k


if __name__ == '__main__':
    d={'a':{'a1':[1,2]}, 'b':{'b1':[3,4], 'b2':[5,6]}}
    ranges = grid_search([], d, [], [])

