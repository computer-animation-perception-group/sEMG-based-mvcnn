class LazyProxy(object):

    def __init__(self, make):
        self._make = make

    def __getattr__(self, name):
        if name == '_inst':
            self._inst = self._make()
            return self._inst
        return getattr(self._inst, name)

    def __setattr__(self, name, value):
        if name in ('_make', '_inst'):
            return super(LazyProxy, self).__setattr__(name, value)
        return setattr(self._inst, name, value)

    def __getstate__(self):
        return self._make

    def __setstate__(self, make):
        self._make = make

    def __hash__(self):
        return hash(self._make)

    def __iter__(self):
        return self._inst.__iter__()
