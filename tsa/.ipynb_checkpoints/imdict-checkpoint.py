class imdict(dict):
    """
    
    Implements immutable dictionary.
    """
    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError('This object is immutable')

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear       = _immutable
    update      = _immutable
    setdefault  = _immutable
    pop         = _immutable
    popitem     = _immutable