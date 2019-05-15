from __future__ import division
import os
from nose.tools import assert_equal
from functools import partial
from logbook import Logger
import mxnet as mx
import numpy as np
import scipy.io as sio
import scipy.stats as sstat
from itertools import product
from collections import OrderedDict
#from lru import LRU
from . import Dataset as Base
from ... import emg_features
from .. import Combo, Trial
from ... import utils, constant
import types
from ...genIndex import genIndex


logger = Logger(__name__)

class FeatureSigImg_ImuData_v2(mx.io.NDArrayIter):

    def __init__(self, *args, **kargs):
        print('Initialization Data Iter')
        self.random_shift_vertical = kargs.pop('random_shift_vertical', 0)
        self.random_shift_horizontal = kargs.pop('random_shift_horizontal', 0)
        self.random_shift_fill = kargs.pop('random_shift_fill', constant.RANDOM_SHIFT_FILL)
        self.amplitude_weighting = kargs.pop('amplitude_weighting', False)
        self.amplitude_weighting_sort = kargs.pop('amplitude_weighting_sort', False)
        self.downsample = kargs.pop('downsample', None)
        self.shuffle = kargs.pop('shuffle', False)
        self.adabn = kargs.pop('adabn', False)
        self._gesture = kargs.pop('gesture')
        self._subject = kargs.pop('subject')
        self._segment = kargs.pop('segment')
        self._index_orig = kargs.pop('index')
        self._index = np.copy(self._index_orig)
        self.num_gesture = kargs.pop('num_gesture')
        self.num_subject = kargs.pop('num_subject')
        self.mini_batch_size = kargs.pop('mini_batch_size', kargs.get('batch_size'))
        self.random_state = kargs.pop('random_state', np.random)
        self.balance_gesture = kargs.pop('balance_gesture', 0)

        self.window = kargs.pop('window', 0)
        self.num_semg_col = kargs.pop('num_semg_col', 0)
        self.num_semg_row = kargs.pop('num_semg_row', 0)
        self.num_imu_col = kargs.pop('num_imu_col', constant.NUM_IMU_COL)
        self.num_imu_row = kargs.pop('num_imu_row', constant.NUM_IMU_ROW)
        super(FeatureSigImg_ImuData_v2, self).__init__(*args, **kargs)

        # feature signal image + imu activity image
        assert (self.data[0][1].shape[3] == self.num_semg_col * self.num_semg_row)
        assert (self.data[1][1].shape[3] == self.num_imu_col * self.num_imu_row)

        self.num_semg_channel = self.data[0][1].shape[2]
        self.num_imu_channel = self.data[1][1].shape[2]

        self.semg_shape1 = len(genIndex(self.num_semg_row * self.num_semg_col)) - 1
        self.semg_shape2 = 1

        self.imu_shape1 = len(genIndex(self.num_imu_row * self.num_imu_col)) - 1
        self.imu_shape2 = 1

        self.num_data = len(self._index)
        self.data_orig = self.data
        assert (self.window == 1)
        self.reset()

    @property
    def num_sample(self):
        return self.num_data

    @property
    def gesture(self):
        return self._gesture[self._index]

    @property
    def subject(self):
        return self._subject[self._index]

    @property
    def segment(self):
        return self._segment[self._index]

    @property
    def provide_data(self):
        assert_equal(len(self.data), 2)
        res = [('semg_data', tuple([self.batch_size, self.num_semg_channel] + list([self.semg_shape1, self.semg_shape2]))),
               ('imu_data', tuple([self.batch_size, self.num_imu_channel] + list([self.imu_shape1, self.imu_shape2])))]
        return res

    def _expand_index(self, index):
        return np.hstack([np.arange(i, i + self.window) for i in index])

    def _reshape_data(self, data):
        return data.reshape(-1, self.window, *data.shape[2:])

    def _get_sigimg(self, data):
        from ... import Context
        import joblib as jb
        res = []

        for amp in Context.parallel(jb.delayed(_get_sigimg_aux)(sample) for sample in data):
            res.append(amp[np.newaxis, ...])
        res = np.concatenate(res, axis=0)
        res = res.reshape(res.shape[0], res.shape[1], res.shape[2], -1)
        return res

    def _get_actimg(self, data):
        from ... import Context
        import joblib as jb
        res = []

        for amp in Context.parallel(jb.delayed(_get_actimg_aux)(sample) for sample in data):
            res.append(amp[np.newaxis, ...])
        res = np.concatenate(res, axis=0)
        res = res.reshape(res.shape[0], res.shape[1], res.shape[2], -1)
        return res

    def _get_segments(self, a, index):
        b = mx.nd.empty((len(index), self.window) + a.shape[2:], dtype=a.dtype)
        for i, j in enumerate(index):
            b[i] = a[j: j + self.window].reshape(self.window, *a.shape[2:])
        return b

    def _getdata(self, data_source):
        """Load data from underlying arrays, internal use only"""
        assert(self.cursor < self.num_data), "DataIter needs reset"

        if data_source is self.data and self.window > 1:
            if self.cursor + self.batch_size <= self.num_data:
                res = [self._get_segments(x[1], self._index[self.cursor:self.cursor + self.batch_size])
                for x in data_source]
            else:
                pad = self.batch_size - self.num_data + self.cursor
                res = [(np.concatenate((self._reshape_data(x[1][self._expand_index(self._index[self.cursor:])]),
                    self._reshape_data(x[1][self._expand_index(self._index[:pad])])), axis=0)) for x in data_source]
        else:
            if self.cursor + self.batch_size <= self.num_data:
                res = [(x[1][self._index[self.cursor:self.cursor + self.batch_size]]) for x in data_source]
            else:
                pad = self.batch_size - self.num_data + self.cursor
                res = [(np.concatenate((x[1][self._expand_index(self._index[self.cursor:])],
                    x[1][self._expand_index(self._index[:pad])]), axis=0)) for x in data_source]

        if data_source is self.data:
            new_res = []
            for a in res:
                new_res.append(a.asnumpy() if isinstance(a, mx.nd.NDArray) else a)
            res = new_res

            res = [a.reshape(a.shape[0], a.shape[2], a.shape[3]) for a in res]
            semg_res = [self._get_sigimg(res[0].asnumpy() if isinstance(res[0], mx.nd.NDArray) else res[0])]
            imu_res = [self._get_actimg(res[1].asnumpy() if isinstance(res[1], mx.nd.NDArray) else res[1])]

            res = semg_res + imu_res
            assert_equal(len(res), 2)

        res = [a if isinstance(a, mx.nd.NDArray) else mx.nd.array(a) for a in res]
        return res

    def _rand(self, smin, smax, shape):
        return (smax - smin) * self.random_state.rand(*shape) + smin

    def _do_shuffle(self):
        if not self.adabn or len(set(self._subject)) == 1:
            self.random_state.shuffle(self._index)
        else:
            batch_size = self.mini_batch_size
            self.random_state.shuffle(self._index)
            subject_shuffled = self._subject[self._index]
            index_batch = []
            for i in sorted(set(self._subject)):
                index = self._index[subject_shuffled == i]
                index = index[:len(index) // batch_size * batch_size]
                index_batch.append(index.reshape(-1, batch_size))
            index_batch = np.vstack(index_batch)
            index = np.arange(len(index_batch))
            self.random_state.shuffle(index)
            self._index = index_batch[index, :].ravel()

            for i in range(0, len(self._subject), batch_size):
                assert np.all(self._subject[self._index[i:i + batch_size - 1]] ==
                    self._subject[self._index[i + 1:i + batch_size]])

            if batch_size != self.batch_size:
                assert self.batch_size % batch_size == 0
                assert (self.batch_size // batch_size) % self.num_subject == 0
                self._index = self._index[:len(self._index) // self.batch_size * self.batch_size].reshape(
                    -1, self.num_subject, batch_size).transpose(0, 2, 1).ravel()

    def reset(self):
        self._reset()
        super(FeatureSigImg_ImuData_v2, self).reset()

    def _reset(self):
        self._index = np.copy(self._index_orig)

        if self.downsample:
            samples = np.arange(len(self._index))
            np.random.RandomState(667).shuffle(samples)
            assert self.downsample > 0 and self.downsample <= 1
            samples = samples[:int(np.round(len(samples) * self.downsample))]
            assert len(samples) > 0
            self._index = self._index[samples]

        if self.balance_gesture:
            num_sample_per_gesture = int(np.round(self.balance_gesture * len(self._index) / self.num_gesture))
            choice = []

            for gesture in set(self.gesture):
                mask = self._gesture[self._index] == gesture
                choice.append(self.random_state.choice(np.where(mask)[0],
                    num_sample_per_gesture))
            choice = np.hstack(choice)
            self._index = self._index[choice]

        if self.shuffle:
            self._do_shuffle()

        self.num_data = len(self._index)


def _get_sigimg_aux(data):
    return np.transpose(get_sig_img(data.T))


def get_sig_img(data):
    ch_num = data.shape[0]
    index = genIndex(ch_num)
    signal_img = data[index]
    signal_img = signal_img[:-1]
    return signal_img


def get_act_img(data):
    signal_img = get_sig_img(data)
    f = np.fft.fft2(signal_img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    return magnitude_spectrum

def _get_actimg_aux(data):
    return np.transpose(get_act_img(data.T))






