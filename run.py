from __future__ import print_function, division
import sys
import os
import numpy as np
import mxnet as mx
from sigr.evaluation import CrossValEvaluation, Exp
from sigr.data import Preprocess, Dataset, Downsample
from sigr import Context


intra_subject_eval = CrossValEvaluation(crossval_type='intra-subject', batch_size=1000)
inter_subject_eval = CrossValEvaluation(crossval_type='inter-subject', batch_size=1000)
one_fold_intra_subject_eval = CrossValEvaluation(crossval_type='one-fold-intra-subject', batch_size=1000)
inter_session_eval = CrossValEvaluation(crossval_type='inter-session', batch_size=1000)
intra_session_eval = CrossValEvaluation(crossval_type='intra-session', batch_size=1000)


def get_accuracies(crossval_type, exps, folds):
    acc = []
    evaluation = CrossValEvaluation(crossval_type=crossval_type, batch_size=1000)
    for fold in folds:
        print(fold)
        acc.append(evaluation.compare(exps, fold))
    return acc


random_state = np.random.RandomState()


def shuf(a):
    b = np.asarray(a).copy()
    random_state.shuffle(b)
    return b


#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'inter-subject',
        #  [Exp(name='S1', dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dbb-inter-subject-%d-v957.40/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))


#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'inter-subject',
        #  [Exp(name='S1', dataset=Dataset.from_name('dbb'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(median)')),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dbb-inter-subject-%d-v957.40.2/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'inter-session',
        #  [Exp(name='S1', dataset=csl, vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7),
                      #  params='.cache/sigr-inter-session-%d-v957.36.5/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'intra-subject',
        #  [Exp(name='S1', dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dbb-intra-subject-%d-v957.42.1/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'inter-session',
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7),
                      #  params='.cache/sigr-inter-session-%d-v957.36.5/model-0028.params')),
         #  Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7),
                      #  params='.cache/sigr-inter-session-%d-v957.36.3/model-0028.params')),
         #  Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7),
                      #  params='.cache/sigr-inter-session-%d-v957.36.4/model-0028.params')),
         #  Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7),
                      #  params='.cache/sigr-inter-session-%d-v957.33/model-0001.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'inter-subject',
        #  [Exp(dataset=Dataset.from_name('dba'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dba-inter-subject-%d-v957.46/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'inter-subject',
        #  [Exp(dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dbc-inter-subject-%d-v957.39/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'inter-subject',
        #  [Exp(dataset=Dataset.from_name('ninapro-db1/5g'), vote=-1,
             #  Mod=dict(num_gesture=5,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10),
                      #  params='.cache/sigr-ninapro-db1-5g-inter-subject-%d-v957.52/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'inter-subject',
        #  [Exp(dataset=Dataset.from_name('ninapro-db1/5g'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('ninapro-peak-50')),
             #  Mod=dict(num_gesture=5,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10),
                      #  params='.cache/sigr-ninapro-db1-5g-inter-subject-%d-v957.52.1/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'inter-subject',
        #  [Exp(dataset=Dataset.from_name('ninapro-db1/5g'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('ninapro-peak-100')),
             #  Mod=dict(num_gesture=5,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10),
                      #  params='.cache/sigr-ninapro-db1-5g-inter-subject-%d-v957.52.2/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'intra-session',
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7),
                      #  params='.cache/sigr-csl-intra-session-%d-v957.51/model-0014.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.8)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'intra-subject',
        #  [Exp(dataset=Dataset.from_name('dba'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dba-intra-subject-%d-v957.49/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dba'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dba-intra-subject-%d-v957.49/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]],
        #  windows=np.arange(1, 301))

#  with Context(parallel=True, level='DEBUG'):
    #  intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dbb-intra-subject-%d-v957.42.1/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]],
        #  windows=np.arange(1, 301))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'intra-subject',
        #  [Exp(name='S1', dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dbc-intra-subject-%d-v957.45/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dbc-intra-subject-%d-v957.45/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]],
        #  windows=np.arange(1, 301))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dba'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dba-inter-subject-%d-v957.46/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]],
        #  windows=np.arange(1, 301))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dbb-inter-subject-%d-v957.40/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]],
        #  windows=np.arange(1, 301))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8),
                      #  params='.cache/sigr-dbc-inter-subject-%d-v957.39/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]],
        #  windows=np.arange(1, 301))

#  with Context(parallel=True, level='DEBUG'):
    #  one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dba'),
             #  Mod=dict(num_gesture=8,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8,
                                        #  num_filter=64),
                      #  params='.cache/sigr-dba-one-fold-intra-subject-%d-v958.3.2/model-0028.params'))],
        #  folds=shuf(range(18)),
        #  windows=np.arange(1, 1001))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7),
                      #  params='.cache/sigr-csl-inter-session-%d-v958.2/model-0028.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])

#  with Context(parallel=True, level='DEBUG'):
    #  one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8,
                                        #  num_filter=64),
                      #  params='.cache/sigr-dbb-one-fold-intra-subject-%d-v958.6.1/model-0028.params'))],
        #  folds=shuf(range(10)),
        #  windows=np.arange(1, 1001))

#  with Context(parallel=True, level='DEBUG'):
    #  one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8,
                                        #  num_filter=64),
                      #  params='.cache/sigr-dbc-one-fold-intra-subject-%d-v958.7.1/model-0028.params'))],
        #  folds=shuf(range(10)),
        #  windows=np.arange(1, 1001))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = get_accuracies(
        #  'inter-session',
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-inter-session-%d-v958.4.1/model-0014.params'))],
        #  folds=[int(f) for f in sys.argv[1:]])
    #  folds = [int(f) for f in sys.argv[1:]]
    #  for i in np.where(np.array(acc) < 0.3)[0]:
        #  print(folds[i], acc[i])
    #  print(np.mean(acc))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dba'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dba-inter-subject-%d-v958.6/model-0028.params'))],
        #  folds=shuf(range(18)),
        #  windows=np.arange(1, 301))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbb-inter-subject-%d-v958.7/model-0028.params'))],
        #  folds=shuf(range(10)),
        #  windows=np.arange(1, 301))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbc-inter-subject-%d-v958.8/model-0028.params'))],
        #  folds=shuf(range(10)),
        #  windows=np.arange(1, 301))

#  with Context(parallel=True, level='DEBUG'):
    #  one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('ninapro-db1/g8'),
             #  Mod=dict(num_gesture=8,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10, num_filter=64),
                      #  params='.cache/sigr-ninapro-db1-g8-one-fold-intra-subject-%d-v958.13.1/model-0028.params'))],
        #  folds=list(range(27)),
        #  windows=np.arange(1, 31))

#  with Context(parallel=True, level='DEBUG'):
    #  one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('ninapro-db1/g12'),
             #  Mod=dict(num_gesture=12,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10, num_filter=64),
                      #  params='.cache/sigr-ninapro-db1-g12-one-fold-intra-subject-%d-v958.12.1/model-0028.params'))],
        #  folds=list(range(27)),
        #  windows=np.arange(1, 31))

#  with Context(parallel=True, level='DEBUG'):
    #  intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.5.1/model-0014.params'))],
        #  folds=shuf(range(250)),
        #  windows=np.arange(1, 2049),
        #  balance=True)


#  with Context(parallel=True, level='DEBUG'):
    #  intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.28.1/model-0014.params'))],
        #  folds=shuf(range(250)),
        #  windows=np.arange(1, 2049),
        #  balance=True)

#  with Context(parallel=True, level='DEBUG'):
    #  intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbb-intra-subject-%d-v958.23.1/model-0028.params'))],
        #  folds=shuf(range(100)),
        #  windows=np.arange(1, 1001))

#  with Context(parallel=True, level='DEBUG'):
    #  intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbc-intra-subject-%d-v958.24.1/model-0028.params'))],
        #  folds=shuf(range(100)),
        #  windows=np.arange(1, 1001))

# ---

#  with Context(parallel=True, level='DEBUG'):
    #  intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbb-intra-subject-%d-v958.14.1/model-0028.params'))],
        #  folds=shuf(range(100)),
        #  windows=np.arange(1, 1001))

#  with Context(parallel=True, level='DEBUG'):
    #  intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbc-intra-subject-%d-v958.15.1/model-0028.params'))],
        #  folds=shuf(range(100)),
        #  windows=np.arange(1, 1001))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-inter-session-%d-v958.17.1/model-0028.params'))],
        #  folds=shuf(range(25)),
        #  windows=np.arange(1, 2049),
        #  balance=True)

#  with Context(parallel=True, level='DEBUG'):
    #  intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7,
                                        #  num_filter=64, num_pixel=0),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.18.1/model-0014.params'))],
        #  folds=shuf(range(250)),
        #  windows=np.arange(1, 2049),
        #  balance=True)

#  with Context(parallel=True, level='DEBUG'):
    #  intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7,
                                        #  num_filter=64, num_pixel=0, num_conv=4),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.19.1/model-0014.params'))],
        #  folds=shuf(range(250)),
        #  windows=np.arange(1, 2049),
        #  balance=True)

#  with Context(parallel=True, level='DEBUG'):
    #  print(int(sys.argv[1]))
    #  for downsample in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        #  print(downsample)
        #  if downsample in (0.001, 0.005):
            #  inter_session_eval = CrossValEvaluation(crossval_type='inter-session', batch_size=100)
        #  else:
            #  inter_session_eval = CrossValEvaluation(crossval_type='inter-session', batch_size=1000)
        #  inter_session_eval.accuracies(
            #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
                 #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                 #  Mod=dict(num_gesture=27,
                          #  adabn=True,
                          #  num_adabn_epoch=10,
                          #  context=[mx.gpu(0)],
                          #  downsample=downsample,
                          #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                          #  params='.cache/sigr-inter-session-%d-v958.4.1.1/model-0014.params'))],
            #  folds=[int(sys.argv[1])])
        #  inter_session_eval.vote_accuracy_curves(
            #  [Exp(dataset=Dataset.from_name('csl'),
                 #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                 #  Mod=dict(num_gesture=27,
                          #  adabn=True,
                          #  num_adabn_epoch=10,
                          #  context=[mx.gpu(0)],
                          #  downsample=downsample,
                          #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                          #  params='.cache/sigr-inter-session-%d-v958.4.1.1/model-0014.params'))],
            #  windows=[1],
            #  balance=True,
            #  folds=[int(sys.argv[1])])

#  print('958.4.1')
#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
                #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                #  Mod=dict(num_gesture=27,
                        #  adabn=True,
                        #  num_adabn_epoch=10,
                        #  context=[mx.gpu(0)],
                        #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                        #  params='.cache/sigr-inter-session-%d-v958.4.1/model-0014.params'))],
        #  folds=shuf(range(25)))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
                #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                #  Mod=dict(num_gesture=27,
                        #  adabn=True,
                        #  num_adabn_epoch=10,
                        #  context=[mx.gpu(0)],
                        #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                        #  params='.cache/sigr-inter-session-%d-v958.4.1/model-0014.params'))],
        #  windows=np.arange(1, 2049),
        #  balance=True,
        #  folds=shuf(range(25)))

if len(sys.argv) > 1:
    print(int(sys.argv[1]))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = intra_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.28.1/model-0014.params'))],
        #  folds=[i for i in range(250) if i not in [39, 197]])
        #  #  folds=[int(sys.argv[1])])
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-inter-session-%d-v958.17.1/model-0028.params'))],
        #  folds=[int(sys.argv[1])])

#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-inter-session-%d-v958.17.1/model-0028.params'))],
        #  folds=[int(sys.argv[1])],
        #  windows=np.arange(1, 2049),
        #  balance=True)

#  print('958.5.1')
#  with Context(parallel=True, level='DEBUG'):
    #  intra_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.5.1/model-0014.params'))],
        #  folds=[int(sys.argv[1])])

#  print('958.5.1 win')
#  with Context(parallel=True, level='DEBUG'):
    #  intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.5.1/model-0014.params'))],
        #  folds=[int(sys.argv[1])],
        #  windows=np.arange(1, 2049),
        #  balance=True)

#  print('958.18.1')
#  with Context(parallel=True, level='DEBUG'):
    #  intra_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7,
                                        #  num_filter=64, num_pixel=0),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.18.1/model-0014.params'))],
        #  folds=[int(sys.argv[1])])

#  print('958.19.1')
#  with Context(parallel=True, level='DEBUG'):
    #  intra_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7,
                                        #  num_filter=64, num_pixel=0, num_conv=4),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.19.1/model-0014.params'))],
        #  folds=[int(sys.argv[1])])

#  print('958.18.1')
#  with Context(parallel=True, level='DEBUG'):
    #  intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7,
                                        #  num_filter=64, num_pixel=0),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.18.1/model-0014.params'))],
        #  folds=[int(sys.argv[1])],
        #  windows=[1],
        #  balance=True)

#  print('958.19.1')
#  with Context(parallel=True, level='DEBUG'):
    #  intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7,
                                        #  num_filter=64, num_pixel=0, num_conv=4),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.19.1/model-0014.params'))],
        #  folds=[int(sys.argv[1])],
        #  windows=[1],
        #  balance=True)

#  ---

#  print('958.18.4')
#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7,
                                        #  num_filter=64, num_pixel=0),
                      #  params='.cache/sigr-csl-inter-session-%d-v958.18.4/model-0014.params'))],
        #  folds=[int(sys.argv[1])])

#  print('958.19.4')
#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7,
                                        #  num_filter=64, num_pixel=0, num_conv=4),
                      #  params='.cache/sigr-csl-inter-session-%d-v958.19.4/model-0014.params'))],
        #  folds=[int(sys.argv[1])])

#  print('958.18.4')
#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7,
                                        #  num_filter=64, num_pixel=0),
                      #  params='.cache/sigr-csl-inter-session-%d-v958.18.4/model-0014.params'))],
        #  folds=[int(sys.argv[1])],
        #  windows=[1],
        #  balance=True)

#  print('958.19.4')
#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7,
                                        #  num_filter=64, num_pixel=0, num_conv=4),
                      #  params='.cache/sigr-csl-inter-session-%d-v958.19.4/model-0014.params'))],
        #  folds=[int(sys.argv[1])],
        #  windows=[1],
        #  balance=True)

#  print('958.5.1')
#  with Context(parallel=True, level='DEBUG'):
    #  intra_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median,peak-256)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.5.1/model-0014.params'))],
        #  folds=[int(sys.argv[1])])

#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
                #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                #  Mod=dict(num_gesture=27,
                        #  adabn=True,
                        #  num_adabn_epoch=10,
                        #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                        #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                        #  params='.cache/sigr-inter-session-%d-v958.4.1/model-0014.params'))],
        #  windows=[1],
        #  balance=True,
        #  folds=[int(sys.argv[1])])

#  with Context(parallel=True, level='DEBUG'):
    #  inter_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbb-inter-subject-%d-v958.7/model-0028.params'))],
        #  folds=[int(sys.argv[1])],
        #  windows=np.arange(1, 1001))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbc-inter-subject-%d-v958.8/model-0028.params'))],
        #  folds=[int(sys.argv[1])],
        #  windows=np.arange(1, 1001))

#  with Context(parallel=True, level='DEBUG'):
    #  intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbc-intra-subject-%d-v958.15.1/model-0028.params'))],
        #  folds=[int(sys.argv[1])],
        #  windows=np.arange(1, 1001))

#  with Context(parallel=True, level='DEBUG'):
    #  if os.path.exists('.cache/sigr-csl-intra-session-%s-v959.1.1/model-0014.params' % sys.argv[1]):
        #  acc = intra_session_eval.accuracies(
            #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
                 #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)'),
                                   #  amplitude_weighting=True),
                 #  Mod=dict(num_gesture=27,
                          #  adabn=True,
                          #  num_adabn_epoch=10,
                          #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                          #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                          #  params='.cache/sigr-csl-intra-session-%d-v959.1.1/model-0014.params'))],
            #  folds=[int(sys.argv[1])])
        #  print(acc)

#  with Context(parallel=True, level='DEBUG'):
    #  folds = []
    #  for i in range(250):
        #  if os.path.exists('.cache/sigr-csl-intra-session-%d-v959.1.1/model-0014.params' % i):
            #  folds.append(i)
    #  folds = shuf(folds)
    #  acc = intra_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)'),
                               #  amplitude_weighting=True),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 2 else int(sys.argv[1]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v959.1.1/model-0014.params'))],
        #  folds=folds)
    #  if np.sum(acc.ravel() < 0.5):
        #  print(folds[acc.ravel() < 0.5])
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = inter_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
                #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                #  Mod=dict(num_gesture=27,
                        #  adabn=True,
                        #  num_adabn_epoch=10,
                        #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                        #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                        #  params='.cache/sigr-inter-session-%d-v959.3.3/model-0014.params'))],
        #  #  folds=[int(sys.argv[1])])
        #  folds=shuf(range(25)))
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('ninapro-db1'),
                #  Mod=dict(num_gesture=52,
                         #  context=[mx.gpu(1)],
                         #  symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10, num_filter=64),
                         #  params='.cache/sigr-ninapro-db1-one-fold-intra-subject-%d-v959.9.1/model-0028.params'))],
        #  #  balance=True,
        #  windows=np.arange(1, 101),
        #  folds=shuf(range(27)))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = inter_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
                #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                #  Mod=dict(num_gesture=27,
                        #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                        #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                        #  params='.cache/sigr-inter-session-%d-v959.4.1/model-0014.params'))],
        #  #  folds=[int(sys.argv[1])])
        #  folds=shuf(range(25)))
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = inter_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
                #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                #  Mod=dict(num_gesture=27,
                        #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                        #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                        #  params='.cache/sigr-inter-session-%d-v959.5.1/model-0014.params'))],
        #  #  folds=[int(sys.argv[1])])
        #  folds=shuf(range(25)))
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = inter_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
                #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                #  Mod=dict(num_gesture=27,
                        #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                        #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                        #  params='.cache/sigr-inter-session-%d-v959.4.2/model-0014.params'))],
        #  #  folds=[int(sys.argv[1])])
        #  folds=shuf(range(25)))
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = inter_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
                #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                #  Mod=dict(num_gesture=27,
                        #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                        #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                        #  params='.cache/sigr-inter-session-%d-v959.4.4/model-0014.params'))],
        #  #  folds=[int(sys.argv[1])])
        #  folds=shuf(range(25)))
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = intra_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,abs,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v959.10.1/model-0014.params'))],
        #  #  folds=[int(sys.argv[1])])
        #  folds=shuf(range(250)))
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,abs,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v959.10.1/model-0014.params'))],
        #  windows=[1],
        #  balance=True,
        #  #  folds=[int(sys.argv[1])])
        #  folds=shuf(range(250)))
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('ninapro-db1/caputo'),
                #  Mod=dict(num_gesture=52,
                         #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                         #  symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10, num_filter=64),
                         #  params='.cache/sigr-ninapro-db1-caputo-one-fold-intra-subject-%d-v959.11.1/model-0028.params'))],
        #  #  balance=True,
        #  windows=np.arange(1, 101),
        #  folds=[int(sys.argv[1])])
        #  #  folds=shuf(range(27)))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbb-inter-session-%d-v959.13.1/model-0014.params'))],
        #  folds=[int(sys.argv[1])],
        #  windows=np.arange(1, 1001))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbb-universal-inter-session-%d-v959.13/model-0028.params' % (int(sys.argv[1]) % 2)))],
        #  folds=[int(sys.argv[1])],
        #  windows=np.arange(1, 1001))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbb-inter-session-%d-v959.13.4/model-0014.params'))],
        #  folds=[int(sys.argv[1])],
        #  windows=np.arange(1, 1001))

#  with Context(parallel=True, level='DEBUG'):
    #  #  if os.path.exists('.cache/sigr-csl-intra-session-%s-v959.1.1/model-0014.params' % sys.argv[1]):
    #  acc = intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)'),
                               #  amplitude_weighting=True,
                               #  amplitude_weighting_sort=True),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v959.1.1/model-0014.params'))],
        #  windows=[307],
        #  balance=True,
        #  folds=shuf(range(250)))
        #  #  folds=[int(sys.argv[1])])
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,rms-307,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v959.12.1/model-0014.params'))],
        #  windows=[1],
        #  balance=True,
        #  folds=shuf([i for i in range(250) if i != 201]))
        #  #  folds=[int(sys.argv[1])])
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = intra_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.5.1.1/model-0028.params'))],
        #  folds=range(250))
        #  #  folds=[int(sys.argv[1])])
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v958.5.1.1/model-0028.params'))],
        #  #  folds=[int(sys.argv[1])],
        #  folds=range(250),
        #  windows=np.arange(1, 2049),
        #  balance=True)

#  with Context(parallel=True, level='DEBUG'):
    #  acc = intra_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)'),
                               #  amplitude_weighting=True,
                               #  amplitude_weighting_sort=True),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v959.1.1.1/model-0028.params'))],
        #  folds=[i for i in range(250) if i not in [99]])
        #  #  folds=[int(sys.argv[1])])
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)'),
                               #  amplitude_weighting=True,
                               #  amplitude_weighting_sort=True),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-csl-intra-session-%d-v959.1.1.1/model-0028.params'))],
        #  windows=[307],
        #  balance=True,
        #  folds=[i for i in range(250) if i not in [99]])
        #  #  folds=[int(sys.argv[1])])
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  val = Dataset.from_name('ninapro-db1/g53').get_one_fold_intra_subject_val(fold=0, batch_size=1000,
                                                                              #  preprocess=Preprocess.parse('ninapro-lowpass'))
    #  total = 0
    #  for batch in val:
        #  total += batch.data[0].shape[0] - batch.pad
    #  print(total)
    #  acc = one_fold_intra_subject_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('ninapro-db1/g53'),
             #  dataset_args=dict(preprocess=Preprocess.parse('ninapro-lowpass')),
             #  Mod=dict(num_gesture=53,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10, num_filter=64),
                      #  params='.cache/sigr-ninapro-db1-g53-one-fold-intra-subject-%d-v958.20.8/model-0028.params'))],
        #  folds=[0])
    #  print(acc)

    #  #  pred, true = one_fold_intra_subject_eval.transform(
        #  #  dataset=Dataset.from_name('ninapro-db1/g53'),
        #  #  dataset_args=dict(preprocess=Preprocess.parse('ninapro-lowpass')),
        #  #  Mod=dict(num_gesture=53,
                 #  #  context=[mx.gpu(0)],
                 #  #  symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10, num_filter=64),
                 #  #  params='.cache/sigr-ninapro-db1-g53-one-fold-intra-subject-%d-v958.20.8/model-0028.params'),
        #  #  fold=0)
    #  #  print(np.argmax(pred, axis=1), true)
    #  #  print(np.sum(np.argmax(pred, axis=1) == true), len(true))

#  with Context(parallel=True, level='DEBUG'):
    #  acc = inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,rms-307,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/sigr-inter-session-%d-v960.3.1/model-0014.params'))],
        #  windows=[1],
        #  balance=True,
        #  folds=shuf([i for i in range(25)]))
        #  #  folds=[int(sys.argv[1])])
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  intra_subject_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('dbb'), vote=-1,
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8,
                                        #  num_filter=64, num_pixel=0),
                      #  params='.cache/sigr-dbb-intra-subject-%d-v960.8.1/model-0028.params'))],
        #  folds=shuf(range(100)))
    #  intra_subject_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('dbb'), vote=-1,
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8,
                                        #  num_filter=64, num_pixel=0, num_conv=4),
                      #  params='.cache/sigr-dbb-intra-subject-%d-v960.14.1/model-0028.params'))],
        #  folds=shuf(range(100)))
    #  inter_subject_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('dbb'), vote=-1,
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8,
                                        #  num_filter=64, num_pixel=0),
                      #  params='.cache/sigr-dbb-inter-subject-%d-v960.10/model-0028.params'))],
        #  folds=shuf(range(10)))
    #  inter_subject_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('dbb'), vote=-1,
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8,
                                        #  num_filter=64, num_pixel=0, num_conv=4),
                      #  params='.cache/sigr-dbb-inter-subject-%d-v960.16/model-0028.params'))],
        #  folds=shuf(range(10)))
    #  intra_subject_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('dbc'), vote=-1,
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8,
                                        #  num_filter=64, num_pixel=0),
                      #  params='.cache/sigr-dbc-intra-subject-%d-v960.9.1/model-0028.params'))],
        #  folds=shuf(range(100)))
    #  intra_subject_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('dbc'), vote=-1,
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8,
                                        #  num_filter=64, num_pixel=0, num_conv=4),
                      #  params='.cache/sigr-dbc-intra-subject-%d-v960.15.1/model-0028.params'))],
        #  folds=shuf(range(100)))
    #  inter_subject_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('dbc'), vote=-1,
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8,
                                        #  num_filter=64, num_pixel=0),
                      #  params='.cache/sigr-dbc-inter-subject-%d-v960.11/model-0028.params'))],
        #  folds=shuf(range(10)))
    #  inter_subject_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('dbc'), vote=-1,
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8,
                                        #  num_filter=64, num_pixel=0, num_conv=4),
                      #  params='.cache/sigr-dbc-inter-subject-%d-v960.17/model-0028.params'))],
        #  folds=shuf(range(10)))
    #  one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('ninapro-db1/caputo'),
                #  Mod=dict(num_gesture=52,
                         #  context=[mx.gpu(0)],
                         #  symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10, num_filter=64,
                                           #  num_pixel=0),
                         #  params='.cache/sigr-ninapro-db1-caputo-one-fold-intra-subject-%d-v960.13.1/model-0028.params'))],
        #  windows=np.arange(1, 101),
        #  folds=shuf(range(27)))
    #  one_fold_intra_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('ninapro-db1/caputo'),
                #  Mod=dict(num_gesture=52,
                         #  context=[mx.gpu(0)],
                         #  symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10, num_filter=64,
                                           #  num_pixel=0, num_conv=4),
                         #  params='.cache/sigr-ninapro-db1-caputo-one-fold-intra-subject-%d-v960.19.1/model-0028.params'))],
        #  windows=np.arange(1, 101),
        #  folds=shuf(range(27)))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
                #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                #  Mod=dict(num_gesture=27,
                         #  context=[mx.gpu(0)],
                         #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                         #  params='.cache/sigr-inter-session-%d-v959.5.1/model-0014.params'))],
        #  balance=True,
        #  windows=np.arange(1, 2049),
        #  folds=[int(sys.argv[1])])
        #  #  folds=shuf(range(25)))

#  with Context(parallel=True, level='DEBUG'):
    #  inter_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
                #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                #  Mod=dict(num_gesture=27,
                         #  context=[mx.gpu(0)],
                         #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                         #  params='.cache/sigr-inter-session-%d-v959.5.1/model-0014.params'))],
        #  folds=[int(sys.argv[1])])

#  with Context(parallel=True, level='DEBUG'):
    #  acc = inter_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbb-inter-subject-%d-v960.21/model-0028.params'))],
        #  #  folds=[int(sys.argv[1])],
        #  folds=np.arange(10),
        #  windows=np.arange(1, 1001))
    #  print(acc.mean(axis=(0, 1))[0])
    #  acc = inter_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbc-inter-subject-%d-v960.22/model-0028.params'))],
        #  #  folds=[int(sys.argv[1])],
        #  folds=np.arange(10),
        #  windows=np.arange(1, 1001))
    #  print(acc.mean(axis=(0, 1))[0])
    #  acc = inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbb-inter-session-%d-v960.23.1/model-0014.params'))],
        #  #  folds=[int(sys.argv[1]) * 2 + 1],
        #  folds=np.arange(10) * 2 + 1,
        #  windows=np.arange(1, 1001))
    #  print(acc.mean(axis=(0, 1))[0])

#  with Context(parallel=True, level='DEBUG'):
    #  acc = inter_subject_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbb-inter-subject-%d-v960.25/model-0028.params'))],
        #  folds=np.arange(10))
        #  #  folds=[int(sys.argv[1])])
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  #  print(int(sys.argv[1]))
    #  gestures = np.arange(27)
    #  np.random.RandomState(621).shuffle(gestures)
    #  print(list(gestures + 1))
    #  for downsample in [gestures[:13]]:
        #  acc = inter_session_eval.accuracies(
            #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
                 #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                 #  Mod=dict(num_gesture=27,
                          #  adabn=True,
                          #  num_adabn_epoch=10,
                          #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                          #  downsample=Downsample.with_gesture_indices(downsample),
                          #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                          #  params='.cache/sigr-inter-session-%d-v958.4.1.1/model-0014.params'))],
            #  #  windows=[1],
            #  #  balance=True,
            #  folds=np.arange(25))
            #  #  folds=[int(sys.argv[1])])
        #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = intra_subject_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbc-intra-subject-%d-v958.24.1/model-0028.params'))],
        #  folds=shuf(range(100)))
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = intra_subject_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  context=[mx.gpu(1)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbb-intra-subject-%d-v958.23.1/model-0028.params'))],
        #  folds=shuf(range(100)))
    #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
                #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
                #  Mod=dict(num_gesture=27,
                         #  adabn=True,
                         #  num_adabn_epoch=10,
                         #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                         #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                         #  params='.cache/sigr-inter-session-%d-v960.24.1/model-0014.params'))],
        #  balance=True,
        #  windows=np.arange(1, 2049),
        #  folds=[int(sys.argv[1])])
        #  #  folds=shuf(range(25)))
    #  #  print(acc.mean())

#  with Context(parallel=True, level='DEBUG'):
    #  acc = inter_subject_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbc'),
             #  Mod=dict(num_gesture=12,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbc-inter-subject-%d-v960.27/model-0028.params'))],
        #  #  folds=[int(sys.argv[1])],
        #  folds=np.arange(10),
        #  windows=np.arange(1, 1001))
    #  print(acc.mean(axis=(0, 1))[0])
    #  acc = inter_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('dbb'),
             #  Mod=dict(num_gesture=8,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0 if len(sys.argv) < 3 else int(sys.argv[2]))],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      #  params='.cache/sigr-dbb-inter-session-%d-v960.26.1/model-0014.params'))],
        #  #  folds=[int(sys.argv[1]) * 2 + 1],
        #  folds=np.arange(10) * 2 + 1,
        #  windows=np.arange(1, 1001))
    #  print(acc.mean(axis=(0, 1))[0])

with Context(parallel=True, level='DEBUG'):
    acc = one_fold_intra_subject_eval.vote_accuracy_curves(
        [Exp(dataset=Dataset.from_name('ninapro-db1/caputo'),
                Mod=dict(num_gesture=52,
                         context=[mx.gpu(0)],
                         symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10, num_filter=64),
                         params='.cache/sigr-ninapro-db1-caputo-one-fold-intra-subject-%d-v960.28.1/model-0028.params'))],
        windows=np.arange(1, 101),
        folds=shuf(range(27)))
    print(acc.mean(axis=(0, 1))[0])
