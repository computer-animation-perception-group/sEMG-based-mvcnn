from __future__ import print_function, division
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import mxnet as mx
from sigr.evaluation import CrossValEvaluation as CV, Exp
from sigr.data import Preprocess, Dataset
from sigr import Context

one_fold_intra_subject_eval = CV(crossval_type='one-fold-intra-subject', batch_size=400)
one_fold_inter_subject_eval = CV(crossval_type='inter-subject', batch_size=1000)
intra_session_eval = CV(crossval_type='intra-session', batch_size=1000)

print('Biopatrec DB1 Singlestream')
print('===========')


semg_row = 8 # sigimg length
semg_col = 1
num_ch =  1131 # feature total channels: 512 + 304 + 308


window=1
num_raw_semg_row=1
num_raw_semg_col=4
feature_name = 'featuresigimg_v2'
fusion_type = 'single'
feature_list = [] # default constant.FEATURE_LIST

with Context(parallel=True, level='DEBUG'):
    acc = one_fold_intra_subject_eval.vote_accuracy_curves(
        [Exp(dataset=Dataset.from_name('biopatrec-db1'),
             Mod=dict(num_gesture=10,
                      context=[mx.gpu(0)],
                      symbol_kargs=dict(dropout=0, fusion_type=fusion_type, num_semg_row=semg_row, num_semg_col=semg_col, num_channel=num_ch, num_filter=64),
                      params='.cache/single-biopatrec-db1-3FSsigimg-win-300-stride-100/one-fold-intra-subject-fold-%d/model-0028.params'))],
        folds=np.arange(17),
        windows=np.arange(1, 5),
        window=window,
        num_semg_row = num_raw_semg_row,
        num_semg_col = num_raw_semg_col,
        feature_name = feature_name,
        balance=True)
    acc = acc.mean(axis=(0, 1))
    print('Single frame accuracy: %f' % acc[0])
