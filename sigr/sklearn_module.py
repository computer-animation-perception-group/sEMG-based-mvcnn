from __future__ import division
from nose.tools import assert_equal
import mxnet as mx
import numpy as np
from logbook import Logger
import joblib as jb
from .base_module import BaseModule


logger = Logger('sigr')


class SklearnModule(BaseModule):

    def _get_data_label(self, data_iter):
        data = []
        label = []
        for batch in data_iter:
            data.append(batch.data[0].asnumpy().reshape(
                batch.data[0].shape[0], -1))
            label.append(batch.label[0].asnumpy())
            if batch.pad:
                data[-1] = data[-1][:-batch.pad]
                label[-1] = label[-1][:-batch.pad]
        data = np.vstack(data)
        label = np.hstack(label)
        assert_equal(len(data), len(label))
        return data, label

    def fit(self, train_data, eval_data, eval_metric='acc', **kargs):
        snapshot = kargs.pop('snapshot')
        self.clf.fit(*self._get_data_label(train_data))
        jb.dump(self.clf, snapshot + '-0001.params')

        if not isinstance(eval_metric, mx.metric.EvalMetric):
            eval_metric = mx.metric.create(eval_metric)
        data, label = self._get_data_label(eval_data)
        pred = self.clf.predict(data).astype(np.int64)
        prob = np.zeros((len(pred), pred.max() + 1))
        prob[np.arange(len(prob)), pred] = 1
        eval_metric.update([mx.nd.array(label)], [mx.nd.array(prob)])
        for name, val in eval_metric.get_name_value():
            logger.info('Epoch[0] Validation-{}={}', name, val)


class KNNModule(SklearnModule):

    def __init__(self):
        from sklearn.neighbors import KNeighborsClassifier as KNN
        self.clf = KNN()

    @classmethod
    def parse(cls, text, **kargs):
        if text == 'knn':
            return cls()


class SVMModule(SklearnModule):

    def __init__(self):
        from sklearn.svm import LinearSVC
        self.clf = LinearSVC()

    @classmethod
    def parse(cls, text, **kargs):
        if text == 'svm':
            return cls()


class RandomForestsModule(SklearnModule):

    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier as RandomForests
        self.clf = RandomForests()

    @classmethod
    def parse(cls, text, **kargs):
        if text == 'random-forests':
            return cls()


class LDAModule(SklearnModule):

    def __init__(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        self.clf = LDA()

    @classmethod
    def parse(cls, text, **kargs):
        if text == 'lda':
            return cls()
