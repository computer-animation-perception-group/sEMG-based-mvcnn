from __future__ import division


class Meta(type):

    impls = []

    def __init__(cls, name, bases, fields):
        type.__init__(cls, name, bases, fields)
        Meta.impls.append(cls)


class BaseModule(object):

    __metaclass__ = Meta

    @classmethod
    def parse(cls, text, **kargs):
        if cls is BaseModule:
            for impl in Meta.impls:
                if impl is not BaseModule:
                    inst = impl.parse(text, **kargs)
                    if inst is not None:
                        return inst


__all__ = ['BaseModule']
