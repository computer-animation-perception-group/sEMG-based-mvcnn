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
config['ninapro-db3'] = dict(
    semg_row = 72,
    semg_col = 1,
    num_raw_semg_row = 1,
    num_raw_semg_col = 12,
    num_ch = 82,
    num_chs = [32, 22, 28],
    num_subject = 6,
    num_gesture = 50,
    window=20,
    stride=1
)
config['ninapro-db4'] = dict(
    semg_row = 72,
    semg_col = 1,
    num_raw_semg_row = 1,
    num_raw_semg_col = 12,
    num_ch = 82,
    num_chs = [32, 22, 28],
    num_subject = 10,
    num_gesture = 52
)
config['ninapro-db6'] = dict(
    semg_row = 98,
    semg_col = 1,
    num_raw_semg_row = 1,
    num_raw_semg_col = 14,
    num_ch = 82,
    num_chs = [32, 22, 28],
    num_subject = 8,
    num_gesture = 7,
    window=20,
    stride=1
)
config['ninapro-db7'] = dict(
    semg_row = 72,
    semg_col = 1,
    num_raw_semg_row = 1,
    num_raw_semg_col = 12,
    num_ch = 82,
    num_chs = [32, 22, 28],
    num_subject = 21,
    num_gesture = 41,
    window=20,
    stride=1
)



one_fold_intra_subject_eval = CV(crossval_type='one-fold-intra-subject', batch_size=1000)
one_fold_inter_subject_eval = CV(crossval_type='inter-subject', batch_size=1000)
four_fold_inter_subject_eval = CV(crossval_type='4-fold-inter-subject', batch_size=1000)
four_fold_intra_subject_eval = CV(crossval_type='4-fold-intra-subject', batch_size=1000)
one_fold_intra_subject_adabn_eval = CV(crossval_type='one-fold-intra-subject-adabn', batch_size=1000)

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
        choices=['single_no_imu', 'single_with_imu',
                 'multi_no_imu', 'multi_with_imu'],
        help='Fusion type',
        default='single_no_imu')
    parser.add_argument(
        '-c',
        '--crossval',
        choices=['one-fold-intra-subject', 'inter-subject',
                 'one-fold-intra-subject-adabn',
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
    elif args.crossval == 'one-fold-intra-subject-adabn':
        eval_type = one_fold_intra_subject_adabn_eval

    dataset_config = config[args.dataset]
    window = 1
    if args.fusion == 'single_no_imu' or args.fusion == 'single_with_imu':
        semg_row = dataset_config['semg_row']
        semg_col = dataset_config['semg_col']
        num_ch = dataset_config['num_ch']
        if args.fusion == 'single_no_imu':
            feature_name = 'featuresigimg_v2'
        elif args.fusion == 'single_with_imu':
            feature_name = 'featuresigimg_imuactimg'
            if args.dataset == 'ninapro-db6' or args.dataset == 'ninapro-db3':
                num_ch = num_ch + 60
            elif args.dataset == 'ninapro-db7':
                num_ch = num_ch + 180
        if args.featurelist == ['tdd_cor']:
            # TSD
            num_ch = 7
        elif args.featurelist == ['dwt']:
            # DWTC
            num_ch = 42
        elif args.featurelist == ['dwpt']:
            # DWPTC
            num_ch = 64
        elif args.featurelist == ['mav', 'wl', 'ssc', 'zc', 'rms', 'mdwt', 'hemg15']:
            # Atzori
            num_ch = 25
        elif args.featurelist == ['iemg', 'var', 'wamp', 'wl', 'ssc']:
            # Du
            num_ch = 5
        elif args.featurelist == ['mav', 'wl', 'ssc', 'zc']:
            # Hudgins
            num_ch = 4
        elif args.featurelist == ['mav', 'wl', 'wamp', 'mavslpframewise', 'arc', 'mnf_MEDIAN_POWER', 'psr']:
            # Phin1
            num_ch = 48
        elif args.featurelist == ['cc', 'rms', 'wl']:
            # Phin2
            num_ch = 6
        elif args.featurelist == ['mav', 'zc', 'ssc', 'wl', 'var', 'wamp', 'arc']:
            # TDAR
            num_ch = 10
        elif args.featurelist == ['cwt']:
            # CWT
            num_ch = 320
        elif args.featurelist == ['hht58', 'arr29', 'mnf', 'mav', 'wl', 'wamp', 'zc', 'mavslpframewise', 'arc', 'mnf_MEDIAN_POWER', 'psr']:
            # Doswald
            num_ch = 251
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
    elif args.fusion == 'multi_with_imu':
        semg_row = []
        semg_col = []
        num_ch = dataset_config['num_chs']
        if args.dataset == 'ninapro-db6' or args.dataset == 'ninapro-db3':
            num_ch.append(60)
        elif args.dataset == 'ninapro-db7':
            num_ch.append(180)

        semg_row.append(dataset_config['semg_row'])
        semg_col.append(dataset_config['semg_col'])

        semg_row.append(dataset_config['semg_row'])
        semg_col.append(dataset_config['semg_col'])

        semg_row.append(dataset_config['semg_row'])
        semg_col.append(dataset_config['semg_col'])

        semg_row.append(dataset_config['semg_row'])
        semg_col.append(dataset_config['semg_col'])
        feature_name = 'rawsemg_feature_imu_multisource_multistream_sigimgv1'

    num_raw_semg_row = dataset_config['num_raw_semg_row']
    num_raw_semg_col = dataset_config['num_raw_semg_col']
    num_gesture = dataset_config['num_gesture']
    num_subject = dataset_config['num_subject']
    W = dataset_config['window']
    S = dataset_config['stride']
    if args.crossval.find('4') != -1:
        num_subject = 4
    fusion_type = args.fusion

    params_path = '.cache/{f}-{c}-flip-{d}-win-{w}-stride-{s}/{c}-%d/model-0028.params'.format(f=fusion_type.replace('_', '-'),
                                                                                              c=args.crossval,
                                                                                              d=args.dataset,
                                                                                              w=W,
                                                                                              s=S)
    if fusion_type.find('single') != -1:
        fusion_type = 'single'

    with Context(parallel=True, level='DEBUG'):
        acc = eval_type.vote_accuracy_curves(
            [Exp(dataset=Dataset.from_name(args.dataset),
                 dataset_args=dict(imu_preprocess=Preprocess.parse('downsample-20')) if args.fusion.find('imu')!=0 else None,
                 Mod=dict(num_gesture=num_gesture,
                          context=[mx.gpu(args.gpu)],
                          multi_stream=True if args.fusion.find('multi') != -1 else False,
                          num_stream=1 if isinstance(num_ch, int) else len(num_ch),
                          symbol_kargs=dict(dropout=0, zscore=True, num_pixel=2, num_stream=len(semg_col), fusion_type=fusion_type, num_semg_row=semg_row, num_semg_col=semg_col, num_channel=num_ch, num_filter=64) if fusion_type.find('multi') != -1 else
                          dict(dropout=0, fusion_type=fusion_type, num_semg_row=semg_row, num_semg_col=semg_col, num_channel=num_ch, num_filter=64),
                          #params='.cache/single-intra-subject-flip-ninapro-db6-with-imu-win-20-stride-1/one-fold-intra-subject-%d/model-0028.params'))],
                          params=params_path))],
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
