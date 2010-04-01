    
def mix_in(OriginalClass, mixInClass):
    """
    This helper function allows to dynamically add a new base class to the given class.
    It is injected at the top of the base class hierarchy.     
    """
    if mixInClass not in OriginalClass.__bases__:
        OriginalClass.__bases__ = (mixInClass,) + OriginalClass.__bases__

def enable_washout(original, washout=0):
    """
    This helper function injects additional code in the given object such
    that during training the first N timesteps are disregarded. This can be applied
    to all trainable nodes, both supervised and unsupervised.
    """
    
    if isinstance(original, type):
        raise Exception('Washout can only be enabled on objects, not classes.')        

    if not hasattr(original, "_train"):
        raise Exception('Object should have a _train method.')

    def _train(x, *args, **kwargs):
        if len(args) > 0:
            original._train_no_washout(x[original.washout:,:], args[0][original.washout:,:], **kwargs)
        else:
            original._train_no_washout(x[original.washout:,:], **kwargs)
    
    setattr(original, "_train_no_washout", original._train)    
    setattr(original, "_train", _train)
    setattr(original, "washout", washout)

