import Oger
import mdp
import numpy as np

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


def optimize_parameters(original_class, gridsearch_parameters=None, cross_validate_function=None, error_measure=None, *args, **kwargs):
    ''' Turn the class original_class into a self-optimizing node, using the given cross-validation and optimization parameters.
    When training the node, it will internally first optimize the parameters given in gridsearch_parameters using the given cross-validation scheme, and 
    then train on the given data using the optimal parameter set.
    optimize_parameters (original_class, gridsearch_parameters=None, cross_validate_function=None, error_measure=None, *args, **kwargs)
    Arguments are:
    - original_class: the node class to change into a self-optimizing node
    - gridsearch_parameters: the dictionary of optimization parameters and corresponding ranges to use. See the documentation for Oger.evaluation.Optimizer for the format
    - cross_validate_function: the function to do cross-validation with
    - error_measure: the loss function based on which the optimal parameters are chosen
    - args/kwargs: additional arguments to be passed to the cross_validate_function
    
    '''
    
    class SelfOptimizingNode():
        def _get_train_seq(self):
            if self._op_is_optimizing:
                return self._op_orig_get_train_seq()
            else:
                return [(self._op_collect_data, self._op_optimize)]
            
        def _op_collect_data(self, x, y):
            self._op_x_list.append(x)
            self._op_y_list.append(y)
              
        def _op_optimize(self):
            self._op_is_optimizing = True
            
            print 'Node ' + str(self) + ' performing self-optimization...'  
            
            # Correct the dictionary key if a copy was made
            if not self in self._op_gridsearch_parameters: 
                if len(self._op_gridsearch_parameters.values()) == 1:
                    self._op_gridsearch_parameters = {self: self._op_gridsearch_parameters.values()[0]}
                else:
                    raise NotImplementedError('There is more than one key in the gridsearch parameter dictionary. Can not correct the dictionary if a copy was made.')
            
            opt = Oger.evaluation.Optimizer(self._op_gridsearch_parameters, loss_function=self._op_error_measure)
                           
            data = [zip(self._op_x_list, self._op_y_list)]
            
            # Do a grid search using the given crossvalidation function and gridsearch parameters
            opt.grid_search(data, flow=mdp.Flow([self, ]), cross_validate_function=self._op_cross_validate_function, *self._op_args, **self._op_kwargs)
            self._op_minimal_error, opt_parameter_dict = opt.get_minimal_error()
            
            # Set the obtained optimal parameter values in the node
            for param in opt_parameter_dict[self]:
                self.__setattr__(param, opt_parameter_dict[self][param])
                print 'Found optimal value for parameter ' + param + ' : ' + str(opt_parameter_dict[self][param]) + ', error: ' + str(self._op_minimal_error)
      
            # Train the node using the optimal parameters
            flow = mdp.Flow([self])
            flow.train(data)
            
            # Empty the list to save memory. Compatible with copies made in other mixins.
            for _ in range(len(self._op_x_list)):
                self._op_x_list.pop()
                self._op_y_list.pop()               
            self._op_is_optimizing = False
        
    
    setattr(original_class, "_op_collect_data", SelfOptimizingNode._op_collect_data)
    setattr(original_class, "_op_optimize", SelfOptimizingNode._op_optimize)
    setattr(original_class, "_op_orig_get_train_seq", original_class._get_train_seq)
    setattr(original_class, "_get_train_seq", SelfOptimizingNode._get_train_seq)
    setattr(original_class, "_op_gridsearch_parameters", gridsearch_parameters)
    setattr(original_class, "_op_cross_validate_function", staticmethod(cross_validate_function))
    setattr(original_class, "_op_error_measure", staticmethod(error_measure))
    setattr(original_class, "_op_args", args)
    setattr(original_class, "_op_kwargs", kwargs)
    setattr(original_class, "_op_is_optimizing", False)
    setattr(original_class, "_op_minimal_error", None)

    setattr(original_class, "_op_x_list", [])
    setattr(original_class, "_op_y_list", [])
    
    # add to base classes
    original_class.__bases__ += (SelfOptimizingNode,)


def select_inputs(original_class, n_inputs=1, error_measure=None):
    ''' Turn the class original_class into a input selection node.
    It selects the inputs using a forward input/feature selection algorithm.
    
    Arguments are:
    - original_class: the node class to change into a self-optimizing node
    - n_inputs: the number of inputs (one input can consist of several signals)
    - error_measure: the loss function based on which the optimal parameters are chosen
                     (is not necessary when optimize_parameters is used)
    
    Cross-validation is not implemented yet, but Oger.utils.optimize_parameters is supported which contains cross-validation...
    First mixin using optimize_parameters and then select_features
    '''
    
    class FeatureSelectionNode():
        def _get_train_seq(self):
            if self._si_selecting_inputs:
                return self._si_orig_get_train_seq()
            else:
                return [(self._si_collect_data, self._si_fwd_input_selection)]
        
        def _si_collect_data(self, x, y):
            self._si_x_list.append(x)
            self._si_y_list.append(y)
              
        def _si_fwd_input_selection(self):
            self._si_selecting_inputs = True
            
            print 'Node ' + str(self) + ' performing feature selection...'
            
            # Rank the features
            n_chans = self._si_x_list[0].shape[1] / self._si_n_inputs             
            self._input_dim = n_chans
            errors = np.zeros((self._si_n_inputs,))
            for i in range(self._si_n_inputs):
                print '\nTesting performance of input', i, '...'
                inps = i * n_chans + np.arange(n_chans)
                x = self._si_remove_data(self._si_x_list, inps)
                node = self.copy()
                flow = mdp.Flow([node, ])
                flow.train([zip(x, self._si_y_list)])
                # Check if optimize parameters was used
                if hasattr(self, '_op_minimal_error'):
                    errors[i] = node._op_minimal_error
                else:
                    errors[i] = self._si_error_measure(flow(x), np.concatenate(tuple(self._si_y_list)))
                node = None
                
            error_indices = self._si_sort_error(errors)
            
            # Add features to the feature set if the error decreases
            self._si_error = errors[error_indices[0]]
            selected_inputs = error_indices[0] * n_chans + np.arange(n_chans)
            for i in range(1, len(error_indices)):
                print '\nTesting combination', i, 'of', len(error_indices) - 1, '...'
                inps = np.concatenate((selected_inputs, error_indices[i] * n_chans + np.arange(n_chans)))
                node = self.copy()
                node._input_dim = len(inps)
                flow = mdp.Flow([node, ])#, self.scheduler) 
                x = self._si_remove_data(self._si_x_list, inps)
                flow.train([zip(x, self._si_y_list)])
                # Check if optimize parameters was used
                if hasattr(self, '_op_minimal_error'):
                    error = node._op_minimal_error
                else:
                    error = self._si_error_measure(flow(x), np.concatenate(tuple(self._si_y_list)))
                node = None    
                # Add to list if error decreases    
                if error < self._si_error:
                    self._si_error = error
                    selected_inputs = inps
            
            # Train the node using the optimal inputs
            print '\nPerform final training...'
            x = self._si_remove_data(self._si_x_list, selected_inputs)
            self._input_dim = len(selected_inputs)
            flow = mdp.Flow([self])#, self.scheduler) 
            flow.train([zip(x, self._si_y_list)])
            
            print '\nFeature selection selected', len(selected_inputs) / n_chans, \
                'input(s) and achieved an error of', self._si_error, '.\n'
            
            # Free memory and set parameters for execution
            self.si_selected_inputs = selected_inputs
            self._input_dim = self._si_n_inputs * n_chans
            for i in range(len(self._si_x_list)):
                self._si_x_list.pop()
                self._si_y_list.pop()               
            self._si_selecting_inputs = False
        
        # Helper function to make datasets
        def _si_remove_data(self, x_list, indices):
            x = []
            for l in x_list:
                x.append(l[:, indices])
            return x
        
        # Helper function to the next best feature   
        def _si_sort_error(self, errors):
            to_big = np.float(np.finfo(errors[0]).max)
            errors_copy = errors.copy()
            errors_copy[np.isnan(errors_copy)] = to_big # ignore nans
            error_indices = []
            for _ in range(len(errors)):
                i = np.argmin(errors_copy)
                if errors_copy[i] == to_big:
                    break
                error_indices.append(i)
                errors_copy[i] = to_big
            return error_indices
        
        def _execute(self, x):
            if self.si_selected_inputs != None:
                return self._si_orig_execute(x[:, self.si_selected_inputs])
            else:
                return self._si_orig_execute(x)   
    
    setattr(original_class, "_si_collect_data", FeatureSelectionNode._si_collect_data)
    setattr(original_class, "_si_fwd_input_selection", FeatureSelectionNode._si_fwd_input_selection)
    setattr(original_class, "_si_orig_get_train_seq", original_class._get_train_seq)
    setattr(original_class, "_get_train_seq", FeatureSelectionNode._get_train_seq)
    setattr(original_class, "_si_selecting_inputs", False)
    setattr(original_class, "_si_orig_execute", original_class._execute)
    setattr(original_class, "_execute", FeatureSelectionNode._execute)
    setattr(original_class, "_si_n_inputs", n_inputs)
    setattr(original_class, "_si_error_measure", staticmethod(error_measure))
    setattr(original_class, "_si_error", None)
    
    setattr(original_class, "_si_remove_data", FeatureSelectionNode._si_remove_data)
    setattr(original_class, "_si_sort_error", FeatureSelectionNode._si_sort_error)
    setattr(original_class, "_si_x_list", [])
    setattr(original_class, "_si_y_list", [])
    setattr(original_class, "si_selected_inputs", None)
    
    # add to base classes
    original_class.__bases__ += (FeatureSelectionNode,)

    
    
def mix_in(washout_class, mixin):
    """
    This helper function allows to dynamically add a new base class to the given class.
    It is injected at the top of the base class hierarchy.     
    """
    if mixin not in washout_class.__bases__:
        washout_class.__bases__ = (mixin,) + washout_class.__bases__
                
def enable_washout(washout_class, washout=0, execute_washout=False):
    """
    This helper function injects additional code in the given class such
    that during training the first N timesteps are disregarded. This can be applied
    to all trainable nodes, both supervised and unsupervised.
    """
    
    if not isinstance(washout_class, type):
        raise Exception('Washout can only be enabled on classes.')        

    if not hasattr(washout_class, "_train"):
        raise Exception('Object should have a _train method.')
    
    if hasattr(washout_class, "washout"):
        raise Exception('Washout already enabled.')

    # helper washout class
    class Washout:
        def _train(self, x, *args, **kwargs):
            if len(args) > 0:
                self._train_no_washout(x[self.washout:, :], args[0][self.washout:, :], **kwargs)
            else:
                self._train_no_washout(x[self.washout:, :], **kwargs)
                
        def _execute(self, x):
            return self._execute_no_washout(x[self.washout:, :])

    # inject new methods
    setattr(washout_class, "_train_no_washout", washout_class._train)    
    setattr(washout_class, "_train", Washout._train)
    setattr(washout_class, "_execute_no_washout", washout_class._execute)
    setattr(washout_class, "washout", washout)        
    if execute_washout:
        setattr(washout_class, "_execute", Washout._execute)


    # add to base classes
    washout_class.__bases__ += (Washout,)

def disable_washout(washout_class):
    """
    Disable previously enabled washout.
    """
    
    if not isinstance(washout_class, type):
        raise Exception('Washout can only be enabled on classes.')        

    if not hasattr(washout_class, "_train"):
        raise Exception('Object should have a _train method.')
    
    if not hasattr(washout_class, "washout"):
        raise Exception('Washout not enabled.')

    del washout_class._train
    del washout_class._execute
    del washout_class.washout
    
    setattr(washout_class, "_train", washout_class._train_no_washout)
    setattr(washout_class, "_execute", washout_class._execute_no_washout)
