from __future__ import print_function, division
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import mxnet as mx
from sigr.evaluation_db1multistream import CrossValEvaluation as CV, Exp
from sigr.data import Preprocess, Dataset
from sigr import Context

one_fold_intra_subject_eval = CV(crossval_type='one-fold-intra-subject', batch_size=1000)
one_fold_inter_subject_eval = CV(crossval_type='inter-subject', batch_size=1000)
intra_session_eval = CV(crossval_type='intra-session', batch_size=1000)

print('NinaPro DB2 Multistream')
print('===========')


semg_row = []
semg_col = []
num_ch = []


num_ch.append(32)
semg_row.append(72)
semg_col.append(1)

num_ch.append(22)
semg_row.append(72)
semg_col.append(1)

num_ch.append(28)
semg_row.append(72)
semg_col.append(1)

#num_ch.append(7)
#semg_row.append(72)
#semg_col.append(1)

#num_ch.append(20)
#semg_row.append(72)
#semg_col.append(1)

window=1
num_raw_semg_row=1
num_raw_semg_col=12
feature_name = 'rawsemg_feature_multisource_multistream_sigimgv2'
fusion_type = 'multi_no_imu'

with Context(parallel=True, level='DEBUG'):
    acc = one_fold_intra_subject_eval.vote_accuracy_curves(
        [Exp(dataset=Dataset.from_name('ninapro-db2'),
             dataset_args=dict(preprocess=Preprocess.parse('ninapro-lowpass,downsample-20')),
             Mod=dict(num_gesture=50,
                      context=[mx.gpu(0)],
                      multi_stream = True,
                      num_stream=len(semg_col),
                      symbol_kargs=dict(dropout=0, zscore=True, num_pixel=2, num_stream=len(semg_col), fusion_type=fusion_type, num_semg_row=semg_row, num_semg_col=semg_col, num_channel=num_ch, num_filter=64),
                      params='.cache/ninapro-db2-multi-downsample20-20-1/one-fold-intra-subject-fold-%d/model-0028.params'))],
        folds=np.arange(40),
        windows=np.arange(1, 5),
        window=window,
        num_semg_row = num_raw_semg_row,
        num_semg_col = num_raw_semg_col,
        feature_name = feature_name,
        balance=True)
    acc = acc.mean(axis=(0, 1))
    print('Single frame accuracy: %f' % acc[0])
