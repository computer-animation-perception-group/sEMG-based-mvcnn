from contextlib import contextmanager
import os
import numpy as np
from .proxy import LazyProxy
assert LazyProxy


@contextmanager
def logging_context(path=None, level=None):
    from logbook import StderrHandler, FileHandler
    from logbook.compat import redirected_logging
    with StderrHandler(level=level or 'INFO').applicationbound():
        if path:
            if not os.path.isdir(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            with FileHandler(path, bubble=True).applicationbound():
                with redirected_logging():
                    yield
        else:
            with redirected_logging():
                yield


def return_list(func):
    import inspect
    from functools import wraps
    assert inspect.isgeneratorfunction(func)

    @wraps(func)
    def wrapped(*args, **kargs):
        return list(func(*args, **kargs))

    return wrapped


@return_list
def continuous_segments(label):
    label = np.asarray(label)

    if not len(label):
        return

    breaks = list(np.where(label[:-1] != label[1:])[0] + 1)
    for begin, end in zip([0] + breaks, breaks + [len(label)]):
        assert begin < end
        yield begin, end


def cached(*args, **kargs):
    import joblib as jb
    from .. import CACHE
    memo = getattr(cached, 'memo', None)
    if memo is None:
        cached.memo = memo = jb.Memory(CACHE, verbose=0)
    return memo.cache(*args, **kargs)


def get_segments(data, window):
    return windowed_view(
        data.flat,
        window * data.shape[1],
        (window - 1) * data.shape[1]
    )


def windowed_view(arr, window, overlap):
    from numpy.lib.stride_tricks import as_strided
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                  window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                   arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)


class Bunch(dict):

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def _packargs(func):
    from functools import wraps
    import inspect

    @wraps(func)
    def wrapped(ctx_or_args, **kargs):
        if isinstance(ctx_or_args, Bunch):
            args = ctx_or_args
        else:
            args = ctx_or_args.obj
        ignore = inspect.getargspec(func).args
        args.update({key: kargs.pop(key) for key in list(kargs)
                     if key not in ignore and key not in args})
        return func(ctx_or_args, **kargs)
    return wrapped


def packargs(func):
    import click
    return click.pass_obj(_packargs(func))


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, data)
    return y


def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cut, fs, order, zero_phase=False):
    from scipy.signal import butter, lfilter, filtfilt

    nyq = 0.5 * fs
    cut = cut / nyq

    b, a = butter(order, cut, btype='low')
    y = (filtfilt if zero_phase else lfilter)(b, a, data)
    return y
