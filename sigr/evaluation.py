from __future__ import division
import os
import numpy as np
from functools import partial
from . import utils
from . import module
from .parse_log import parse_log
from logbook import Logger
from copy import deepcopy
import mxnet as mx


Exp = utils.Bunch


logger = Logger(__name__)


@utils.cached(ignore=['context'])
def _crossval_predict_aux(self, Mod, get_crossval_val, fold, context,
                          feature_name,
                          window,
                          num_semg_row,
                          num_semg_col,
                          feature_list=[],
                          dataset_args=None):
    Mod = deepcopy(Mod)
    Mod.update(context=context)
    mod = module.RuntimeModule(**Mod)
    Val = partial(
        get_crossval_val,
        fold=fold,
        batch_size=self.batch_size,
        window=window,
        feature_name=feature_name,
        num_semg_row=num_semg_row,
        num_semg_col=num_semg_col,
        feature_list=feature_list,
        **(dataset_args or {})
    )
#    print Val.name
    return mod.predict(utils.LazyProxy(Val))


@utils.cached(ignore=['context'])
def _crossval_predict_proba_aux(self, Mod, get_crossval_val, fold, context,
                                feature_name,
                                window,
                                num_semg_row,
                                num_semg_col,
                                feature_list=[],
                                dataset_args=None):
    Mod = deepcopy(Mod)
    Mod.update(context=context)
    mod = module.RuntimeModule(**Mod)
    Val = partial(
        get_crossval_val,
        fold=fold,
        batch_size=self.batch_size,
        window=window,
        feature_name=feature_name,
        num_semg_row=num_semg_row,
        num_semg_col=num_semg_col,
        feature_list=feature_list,
        **(dataset_args or {})
    )
    return mod.predict_proba(utils.LazyProxy(Val))


def _crossval_predict(self, **kargs):
    proba = kargs.pop('proba', False)
    fold = int(kargs.pop('fold'))
    Mod = kargs.pop('Mod')
    Mod = deepcopy(Mod)
    Mod.update(params=self.format_params(Mod['params'], fold))
    context = Mod.pop('context', [mx.gpu(0)])
    window = kargs.pop('window')
    feature_name = kargs.pop('feature_name')
    num_semg_row=kargs.pop('num_semg_row')
    num_semg_col=kargs.pop('num_semg_col')
    feature_list=kargs.pop('feature_list', [])
    #  import pickle
    #  d = kargs.copy()
    #  d.update(Mod=Mod, fold=fold)
    #  print(pickle.dumps(d))

    #  Ensure load from disk.
    #  Otherwise following cached methods like vote will have two caches,
    #  one for the first computation,
    #  and the other for the cached one.
    func = _crossval_predict_aux if not proba else _crossval_predict_proba_aux
    return func.call_and_shelve(self, Mod=Mod, fold=fold, context=context,
                                window=window, feature_name=feature_name,
                                num_semg_row=num_semg_row, num_semg_col=num_semg_col,
                                feature_list=feature_list, **kargs).get()


class Evaluation(object):

    def __init__(self, batch_size=None):
        self.batch_size = batch_size


class CrossValEvaluation(Evaluation):

    def __init__(self, **kargs):
        self.crossval_type = kargs.pop('crossval_type')
        super(CrossValEvaluation, self).__init__(**kargs)

    def get_crossval_val_func(self, dataset):
        return getattr(dataset, 'get_%s_val' % self.crossval_type.replace('-', '_'))

    def format_params(self, params, fold):
        try:
            return params % fold
        except:
            return params

    def transform(self, Mod, dataset, fold, dataset_args=None):
        get_crossval_val = self.get_crossval_val_func(dataset)
        pred, true, _ = _crossval_predict(
            self,
            proba=True,
            Mod=Mod,
            get_crossval_val=get_crossval_val,
            fold=fold,
            dataset_args=dataset_args)
        return pred, true

    def accuracy_mod(self, Mod, dataset, fold,
                     vote=False,
                     dataset_args=None,
                     balance=False):
        get_crossval_val = self.get_crossval_val_func(dataset)
        pred, true, segment = _crossval_predict(
            self,
            Mod=Mod,
            get_crossval_val=get_crossval_val,
            fold=fold,
            dataset_args=dataset_args)
        if vote:
            from .vote import vote as do
            return do(true, pred, segment, vote, balance)
        return (true == pred).sum() / true.size

    def accuracy_exp(self, exp, fold):
        if hasattr(exp, 'Mod') and hasattr(exp, 'dataset'):
            return self.accuracy_mod(Mod=exp.Mod,
                                     dataset=exp.dataset,
                                     fold=fold,
                                     vote=exp.get('vote', False),
                                     dataset_args=exp.get('dataset_args'))
        else:
            try:
                return parse_log(os.path.join(exp.root % fold, 'log')).val.iloc[-1]
            except:
                return np.nan

    def accuracy(self, **kargs):
        if 'exp' in kargs:
            return self.accuracy_exp(**kargs)
        elif 'Mod' in kargs:
            return self.accuracy_mod(**kargs)
        else:
            assert False

    def accuracies(self, exps, folds):
        acc = []
        for exp in exps:
            for fold in folds:
                acc.append(self.accuracy(exp=exp, fold=fold))
        return np.array(acc).reshape(len(exps), len(folds))

    def compare(self, exps, fold):
        acc = []
        for exp in exps:
            if hasattr(exp, 'Mod') and hasattr(exp, 'dataset'):
                acc.append(self.accuracy(Mod=exp.Mod,
                                         dataset=exp.dataset,
                                         fold=fold,
                                         vote=exp.get('vote', False),
                                         dataset_args=exp.get('dataset_args')))
            else:
                try:
                    acc.append(parse_log(os.path.join(exp.root % fold, 'log')).val.iloc[-1])
                except:
                    acc.append(np.nan)
        return acc

    def vote_accuracy_curves(self, exps, folds, windows, feature_name, window, num_semg_row, num_semg_col, feature_list=[], balance=False):
        acc = []
        for exp in exps:
            for fold in folds:
                acc.append(self.vote_accuracy_curve(
                    Mod=exp.Mod,
                    dataset=exp.dataset,
                    fold=int(fold),
                    windows=windows,
                    feature_name=feature_name,
                    window=window,
                    num_semg_row=num_semg_row,
                    num_semg_col=num_semg_col,
                    feature_list=feature_list,
                    dataset_args=exp.get('dataset_args'),
                    balance=balance))
        return np.array(acc).reshape(len(exps), len(folds), len(windows))

    def vote_accuracy_curve(self, Mod, dataset, fold, windows, feature_name, window, num_semg_row, num_semg_col,
                            feature_list=[],
                            dataset_args=None,
                            balance=False):
        get_crossval_val = self.get_crossval_val_func(dataset)
        pred, true, segment = _crossval_predict(
            self,
            Mod=Mod,
            get_crossval_val=get_crossval_val,
            fold=fold,
            feature_name=feature_name,
            window = window,
            num_semg_row = num_semg_row,
            num_semg_col = num_semg_col,
            feature_list=feature_list,
            dataset_args=dataset_args)
        from .vote import get_vote_accuracy_curve as do
        return do(true, pred, segment, windows, balance)[1]


def get_crossval_accuracies(crossval_type, exps, folds, batch_size=1000):
    acc = []
    evaluation = CrossValEvaluation(
        crossval_type=crossval_type,
        batch_size=batch_size
    )
    for fold in folds:
        acc.append(evaluation.compare(exps, fold))
    return acc
