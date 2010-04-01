  
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

    # helper washout class
    class Washout:
        def _train(self, x, *args, **kwargs):
            if len(args) > 0:
                self._train_no_washout(x[self.washout:,:], args[0][self.washout:,:], **kwargs)
            else:
                self._train_no_washout(x[self.washout:,:], **kwargs)
                
    # inject new methods
    setattr(original, "_train_no_washout", original._train)    
    setattr(original, "_train", Washout._train)
    setattr(original, "washout", washout)

    # add to base classes
    original.__bases__ += (Washout,)
