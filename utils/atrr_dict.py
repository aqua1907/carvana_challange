class AttrDict(dict):
    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        elif item in self:
            return self[item]
        else:
            AttributeError(item)

    def __setattr__(self, key, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if key in self.__dict__:
                self.__dict__[key] = value
            else:
                self[key] = value
        else:
            raise AttributeError('Attempted to set "{}" to "{}", but AttrDict is immutable'.
                                 format(key, value))

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]
