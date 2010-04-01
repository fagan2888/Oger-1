    
def MixIn(OriginalClass, mixInClass):
    """
    This helper function allows to dynamically add a new base class to the given class.
    It is injected at the top of the base class hierarchy.     
    """
    if mixInClass not in OriginalClass.__bases__:
        OriginalClass.__bases__ = (mixInClass,) + OriginalClass.__bases__

def EnableWashout(OriginalClass, washout=0):
    """
    This helper function injects additional code in the given class or object such
    that during training the first N timesteps are disregarded. This can be applied
    to all trainable nodes, both supervised and unsupervised.
    """
    def _train(self, x, *args, **kwargs):
        if len(args) > 0:
            self._train_no_washout(x[self.washout:,:], args[0][self.washout:,:], **kwargs)
        else:
            self._train_no_washout(x[self.washout:,:], **kwargs)
    
    setattr(OriginalClass, "_train_no_washout", OriginalClass._train)    
    setattr(OriginalClass, "_train", _train)
    setattr(OriginalClass, "washout", washout)

