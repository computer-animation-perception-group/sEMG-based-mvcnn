from __future__ import print_function, division
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import mxnet as mx
from sigr.evaluation import CrossValEvaluation as CV, Exp
from sigr.data import Preprocess, Dataset
from sigr import Context
import argparse

config = {}
config['ninapro-db1'] = dict(
    semg_row = 50,
    semg_col = 1,
    num_raw_semg_row=1,
    num_raw_semg_col=10,
    num_ch = 82
)

config['biopatrec-db1'] = dict(
    semg_row = 8,
    semg_col = 1,
    num_raw_semg_row = 1,
    num_raw_semg_col = 4,
    num_ch = 1131,
    num_chs = [512, 304, 315],
    num_gesture = 10,
    num_subject = 20
)
config['biopatrec-db2'] = dict(
    semg_row = 32,
    semg_col = 1,
    num_raw_semg_row = 1,
    num_raw_semg_col = 8,
    num_ch = 1131,
    num_chs = [512, 304, 315],
    num_gesture = 26,
    num_subject = 17
)
config['biopatrec-db3'] = dict(
    semg_row = 8,
    semg_col = 1,
    num_raw_semg_row = 1,
    num_raw_semg_col = 4,
    num_ch = 1131,
    num_chs = [512, 304, 315],
    num_gesture = 10,
    num_subject = 8
)
config['biopatrec-db4'] = dict(
    semg_row = 128,
    semg_col = 1,
    num_raw_semg_row = 1,
    num_raw_semg_col = 16,
    num_ch = 1131,
    num_chs = [512, 304, 315],
    num_gesture = 8,
    num_subject = 8
)



one_fold_intra_subject_eval = CV(crossval_type='one-fold-intra-subject', batch_size=800)
one_fold_inter_subject_eval = CV(crossval_type='inter-subject', batch_size=1000)
four_fold_inter_subject_eval = CV(crossval_type='4-fold-inter-subject', batch_size=1000)
four_fold_intra_subject_eval = CV(crossval_type='4-fold-intra-subject', batch_size=1000)
intra_session_eval = CV(crossval_type='intra-session', batch_size=1000)

def main():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        '-d',
        '--dataset',
        choices=['ninapro-db1', 'ninapro-db2', 'ninapro-db3', 'ninapro-db4',
                 'ninapro-db5', 'ninapro-db6', 'ninapro-db7', 'biopatrec-db1',
                 'biopatrec-db2', 'biopatrec-db3', 'biopatrec-db4'],
        help='Dataset choices',
        required=True)
    parser.add_argument(
        '--fusion',
        choices=['single', 'multi_no_imu', 'multi_with_imu'],
        help='Fusion type',
        default='single')
    parser.add_argument(
        '-c',
        '--crossval',
        choices=['one-fold-intra-subject', 'inter-subject',
                 '4-fold-inter-subject', '4-fold-intra-subject'],
        default='one-fold-intra-subject',
        help='Crossval type')
    parser.add_argument(
        '-f',
        '--featurelist',
        nargs='+',
        help='Features to use, None means using constant.Feature_list')
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='Gpu to use')

    args = parser.parse_args()

    run(args)


def run(args):

    print('{0}-{1}-{2}-{3}'.format(args.dataset, args.fusion, args.crossval, args.featurelist))
    print('=======================')

    if args.crossval == 'one-fold-intra-subject':
        eval_type = one_fold_intra_subject_eval
    elif args.crossval == 'inter-subject':
        eval_type = one_fold_inter_subject_eval
    elif args.crossval == '4-fold-inter-subject':
        eval_type = four_fold_inter_subject_eval
    elif args.crossval == '4-fold-intra-subject':
        eval_type = four_fold_intra_subject_eval

    dataset_config = config[args.dataset]
    window = 1
    if args.fusion == 'single':
        semg_row = dataset_config['semg_row']
        semg_col = dataset_config['semg_col']
        num_ch = dataset_config['num_ch']
        feature_name = 'featuresigimg_v2'
        if args.featurelist == ['tdd_cor']:
            # TSD
            num_ch = 7
        elif args.featurelist == ['dwt']:
            # DWTC
            num_ch = 304
        elif args.featurelist == ['dwpt']:
            # DWPTC
            num_ch = 512
        elif args.featurelist == ['mav', 'wl', 'ssc', 'zc', 'rms', 'mdwt', 'hemg15']:
            # Atzori
            num_ch = 28
        elif args.featurelist == ['iemg', 'var', 'wamp', 'wl', 'ssc', 'zc']:
            # Du
            num_ch = 6
        elif args.featurelist == ['mav', 'wl', 'ssc', 'zc']:
            # Hudgins
            num_ch = 4
        elif args.featurelist == ['mav', 'wl', 'wamp', 'zc', 'mavslpframewise', 'arc', 'mnf_MEDIAN_POWER', 'psr']:
            # Phin1
            num_ch = 309
        elif args.featurelist == ['sampen', 'cc', 'rms', 'wl']:
            # Phin2
            num_ch = 9
        elif args.featurelist == ['mav', 'zc', 'ssc', 'wl', 'var', 'wamp', 'arc']:
            # TDAR
            num_ch = 10
        elif args.featurelist == ['cwt']:
            # CWT
            num_ch = 2400
        elif args.featurelist == ['hht58', 'arr29', 'mnf', 'mav', 'wl', 'wamp', 'zc', 'mavslpframewise', 'arc', 'mnf_MEDIAN_POWER', 'psr']:
            # Doswald
            num_ch = 1291
    elif args.fusion == 'multi_no_imu':
        semg_row = []
        semg_col = []
        num_ch = dataset_config['num_chs']

        semg_row.append(dataset_config['semg_row'])
        semg_col.append(dataset_config['semg_col'])

        semg_row.append(dataset_config['semg_row'])
        semg_col.append(dataset_config['semg_col'])

        semg_row.append(dataset_config['semg_row'])
        semg_col.append(dataset_config['semg_col'])
        feature_name = 'rawsemg_feature_multisource_multistream_sigimgv2'

    num_raw_semg_row = dataset_config['num_raw_semg_row']
    num_raw_semg_col = dataset_config['num_raw_semg_col']
    num_gesture = dataset_config['num_gesture']
    num_subject = dataset_config['num_subject']
    if args.crossval.find('4') != -1:
        num_subject = 4
    fusion_type = args.fusion

    with Context(parallel=True, level='DEBUG'):
        acc = eval_type.vote_accuracy_curves(
            [Exp(dataset=Dataset.from_name(args.dataset),
                 Mod=dict(num_gesture=num_gesture,
                          context=[mx.gpu(args.gpu)],
                          multi_stream=True if args.fusion.find('multi') != -1 else False,
                          num_stream=1 if isinstance(num_ch, int) else len(num_ch),
                          symbol_kargs=dict(dropout=0, zscore=True, num_pixel=2, num_stream=len(semg_col), fusion_type=fusion_type, num_semg_row=semg_row, num_semg_col=semg_col, num_channel=num_ch, num_filter=64) if fusion_type.find('multi') != -1 else
                          dict(dropout=0, fusion_type=fusion_type, num_semg_row=semg_row, num_semg_col=semg_col, num_channel=num_ch, num_filter=64),
                          params='.cache/multi-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/dropout0.7-wd0.001/4-fold-inter-subject-%d/model-0028.params'))],
            folds=np.arange(num_subject),
            windows=np.arange(1, 5),
            window=window,
            num_semg_row = num_raw_semg_row,
            num_semg_col = num_raw_semg_col,
            feature_name = feature_name,
            feature_list = [] if args.featurelist is None else args.featurelist,
            balance=True)
        acc = acc.mean(axis=(0, 1))
        print('Single frame accuracy: %f' % acc[0])


if __name__ == '__main__':
    main()
