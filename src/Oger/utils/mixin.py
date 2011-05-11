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
        print ('Warning: washout already enabled.')
        return

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
