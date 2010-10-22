import Oger
import mdp

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


def optimize_parameters(original, gridsearch_parameters=None, cross_validate_function=None, error_measure=None, *args, **kwargs):
    ''' Turn the class original into a self-optimizing node, using the given cross-validation and optimization parameters.
    When training the node, it will internally first optimize the parameters given in gridsearch_parameters using the given cross-validation scheme, and 
    then train on the given data using the optimal parameter set.
    optimize_parameters (original, gridsearch_parameters=None, cross_validate_function=None, error_measure=None, *args, **kwargs)
    Arguments are:
    - original: the node class to change into a self-optimizing node
    - gridsearch_parameters: the dictionary of optimization parameters and corresponding ranges to use. See the documentation for Oger.evaluation.Optimizer for the format
    - cross_validate_function: the function to do cross-validation with
    - error_measure: the loss function based on which the optimal parameters are chosen
    - args/kwargs: additional arguments to be passed to the cross_validate_function
    
    '''
    
    class SelfOptimizingNode():
        def _get_train_seq(self):
            if self._po_is_optimizing:
                return self._po_orig_get_train_seq()
            else:
                return [(self._po_collect_data, self._po_optimize)]
            
        def _po_collect_data(self, x, y):
            self._po_x_list.append(x)
            self._po_y_list.append(y)
              
        def _po_optimize(self):
            self._po_is_optimizing = True
            
            print 'Node ' + str(self) + ' performing self-optimization...'   
            opt = Oger.evaluation.Optimizer(self._po_gridsearch_parameters, self._po_error_measure)
                           
            data = [zip(self._po_x_list, self._po_y_list)]
            
            # Do a grid search using the given crossvalidation function and gridsearch parameters
            opt.grid_search(data, flow=mdp.Flow([self, ]), cross_validate_function=self._po_cross_validate_function, *self._po_args, **self._po_kwargs)
            _, opt_parameter_dict = opt.get_minimal_error()
            
            # Set the obtained optimal parameter values in the node
            for param in opt_parameter_dict[self]:
                self.__setattr__(param, opt_parameter_dict[self][param])
                print 'Found optimal value for parameter ' + param + ' : ' + str(opt_parameter_dict[self][param])
            
            flow = mdp.Flow([self])
            flow.train(data)
                
            self._po_is_optimizing = False
        
    
    setattr(original, "_po_collect_data", SelfOptimizingNode._po_collect_data)
    setattr(original, "_po_optimize", SelfOptimizingNode._po_optimize)
    setattr(original, "_po_orig_get_train_seq", original._get_train_seq)
    setattr(original, "_get_train_seq", SelfOptimizingNode._get_train_seq)
    setattr(original, "_po_gridsearch_parameters", gridsearch_parameters)
    setattr(original, "_po_cross_validate_function", staticmethod(cross_validate_function))
    setattr(original, "_po_error_measure", staticmethod(error_measure))
    setattr(original, "_po_args", args)
    setattr(original, "_po_kwargs", kwargs)
    setattr(original, "_po_is_optimizing", False)

    setattr(original, "_po_x_list", [])
    setattr(original, "_po_y_list", [])
    
    # add to base classes
    original.__bases__ += (SelfOptimizingNode,)

    
    
def mix_in(original, mixin):
    """
    This helper function allows to dynamically add a new base class to the given class.
    It is injected at the top of the base class hierarchy.     
    """
    if mixin not in original.__bases__:
        original.__bases__ = (mixin,) + original.__bases__
                
def enable_washout(original, washout=0):
    """
    This helper function injects additional code in the given class such
    that during training the first N timesteps are disregarded. This can be applied
    to all trainable nodes, both supervised and unsupervised.
    """
    
    if not isinstance(original, type):
        raise Exception('Washout can only be enabled on classes.')        

    if not hasattr(original, "_train"):
        raise Exception('Object should have a _train method.')
    
    if hasattr(original, "washout"):
        raise Exception('Washout already enabled.')

    # helper washout class
    class Washout:
        def _train(self, x, *args, **kwargs):
            if len(args) > 0:
                self._train_no_washout(x[self.washout:, :], args[0][self.washout:, :], **kwargs)
            else:
                self._train_no_washout(x[self.washout:, :], **kwargs)
                
    # inject new methods
    setattr(original, "_train_no_washout", original._train)    
    setattr(original, "_train", Washout._train)
    setattr(original, "washout", washout)

    # add to base classes
    original.__bases__ += (Washout,)
