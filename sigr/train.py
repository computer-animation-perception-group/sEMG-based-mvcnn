from __future__ import division
import click
import mxnet as mx
from logbook import Logger
from pprint import pformat
import os
from .utils import packargs, Bunch
#from .module import Module
from .module import Module
from .data import Preprocess, Dataset
from . import Context, constant
import types
import ast

class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


logger = Logger('semgfeature')


@click.group()
def cli():
    pass


@cli.group()
@click.option('--downsample', type=int, default=0)
@click.option('--num-epoch', type=int, default=60, help='Maximum epoches')
@click.option('--num-tzeng-batch', type=int, default=constant.NUM_TZENG_BATCH,
              help='Batch number of each Tzeng update')
@click.option('--lr-step', type=int, multiple=True, default=[20, 40], help='Epoch numbers to decay learning rate')
@click.option('--lr-factor', type=float, multiple=True)
@click.option('--lr', type=float, default=0.1, help='Base learning rate')
@click.option('--wd', type=float, default=0.0001, help='Weight decay')
@click.option('--subject-wd', type=float, help='Weight decay multipler of the subject branch')
@click.option('--gamma', type=float, default=constant.GAMMA, help='Gamma in RevGrad')
@click.option('--revgrad', is_flag=True, help='Use RevGrad')
@click.option('--num-revgrad-batch', type=int, default=2, help='Batch number of each RevGrad update')
@click.option('--tzeng', is_flag=True, help='Use Tzeng_ICCV_2015')
@click.option('--confuse-conv', is_flag=True, help='Domain confusion (fro both RevGrad and Tzeng) on conv2')
@click.option('--confuse-all', is_flag=True, help='Domain confusion (for borh RevGrad and Tzeng) on all layers')
@click.option('--subject-loss-weight', type=float, default=1, help='Ganin et al. use 0.1 in their code')
@click.option('--subject-confusion-loss-weight', type=float, default=1,
              help='Tzeng confusion loss weight, larger than 1 seems better')
@click.option('--lambda-scale', type=float, default=constant.LAMBDA_SCALE,
              help='Global scale of lambda in RevGrad, 1 in their paper and 0.1 in their code')
@click.option('--params', type=click.Path(exists=True), help='Inital weights')
@click.option('--ignore-params', multiple=True, help='Ignore params in --params with regex')
@click.option('--num-feature-block', type=int, default=constant.NUM_FEATURE_BLOCK, help='Number of FC layers in feature extraction part')
@click.option('--num-gesture-block', type=int, default=constant.NUM_GESTURE_BLOCK, help='Number of FC layers in gesture branch')
@click.option('--num-subject-block', type=int, default=constant.NUM_SUBJECT_BLOCK, help='Number of FC layers in subject branch')
@click.option('--num-pixel', type=int, default=constant.NUM_PIXEL, help='Pixelwise reduction layers')
@click.option('--num-filter', type=int, default=constant.NUM_FILTER, help='Kernels of the conv layers')
@click.option('--num-hidden', type=int, default=constant.NUM_HIDDEN, help='Kernels of the FC layers')
@click.option('--num-bottleneck', type=int, default=constant.NUM_BOTTLENECK, help='Kernels of the bottleneck layer')
@click.option('--dropout', type=float, default=constant.DROPOUT, help='Dropout ratio')
@click.option('--num-presnet', type=int, multiple=True, help='Deprecated')
@click.option('--presnet-branch', type=int, multiple=True, help='Deprecated')
@click.option('--drop-presnet', is_flag=True)
@click.option('--bng', is_flag=True, help='Deprecated')
@click.option('--drop-branch', is_flag=True, help='Dropout after each FC in branches')
@click.option('--pool', is_flag=True, help='Deprecated')
@click.option('--fft', is_flag=True, help='Deprecaded. Perform FFT and use spectrum amplitude as image channels. Cannot be used on non-uniform (segment length) dataset like NinaPro')
@click.option('--fft-append', is_flag=True, help='Append FFT feature to raw frames in channel axis')
@click.option('--dual-stream', is_flag=True, help='Use raw frames and FFT feature as dual-stream')
@click.option('--zscore/--no-zscore', default=True, help='Use z-score normalization on input')
@click.option('--zscore-bng', is_flag=True, help='Use global BatchNorm as z-score normalization, for window > 1 or FFT')
@click.option('--dense-window/--no-dense-window', default=True, help='Dense sampling of windows during training')
@click.option('--faug', type=float, default=0)
@click.option('--faug-classwise', is_flag=True)
@click.option('--num-eval-epoch', type=int, default=1)
@click.option('--snapshot-period', type=int, default=1)
@click.option('--drop-conv', is_flag=True)
@click.option('--drop-pixel', type=int, multiple=True, default=(-1,))
@click.option('--drop-presnet-branch', is_flag=True)
@click.option('--drop-presnet-proj', is_flag=True)
@click.option('--fix-params', multiple=True)
@click.option('--presnet-proj-type', type=click.Choice(['A', 'B']), default='A')
@click.option('--decay-all', is_flag=True)
@click.option('--presnet-promote', is_flag=True)
@click.option('--pixel-reduce-loss-weight', type=float, default=0)
@click.option('--fast-pixel-reduce/--no-fast-pixel-reduce', default=True)
@click.option('--pixel-reduce-bias', is_flag=True)
@click.option('--pixel-reduce-kernel', type=int, multiple=True, default=(1, 1))
@click.option('--pixel-reduce-stride', type=int, multiple=True, default=(1, 1))
@click.option('--pixel-reduce-pad', type=int, multiple=True, default=(0, 0))
@click.option('--pixel-reduce-norm', is_flag=True)
@click.option('--pixel-reduce-reg-out', is_flag=True)
@click.option('--num-pixel-reduce-filter', type=int, multiple=True, default=tuple(None for _ in range(constant.NUM_PIXEL)))
@click.option('--num-conv', type=int, default=2)
@click.option('--pixel-same-init', is_flag=True)
@click.option('--presnet-dense', is_flag=True)
@click.option('--conv-shortcut', is_flag=True)
@click.option('--bandstop', is_flag=True)
@click.option('--gpu', type=int, multiple=True, default=[0])
@click.option('--gpu-x', type=int, default=0)
@click.option('--log', type=click.Path(), help='Path of logging file')
@click.option('--snapshot', type=click.Path(), help='Snapshot prefix')
@click.option('--root', type=click.Path(), help='Root path of the experiment, auto create if not exists')
@click.option('--num-epoch', type=int, default=60, help='Maximum epochs')
@click.option('--num-semg-row', type=int, default=constant.NUM_SEMG_ROW, help='Rows of sEMG image')
@click.option('--num-semg-col', type=int, default=constant.NUM_SEMG_COL, help='Cols of sEMG image')
@click.option('--num-imu-row', type=int, default=constant.NUM_IMU_ROW, help='Rows of IMU image')
@click.option('--num-imu-col', type=int, default=constant.NUM_IMU_COL, help='Cols of IMU image')
@click.option('--random-shift-fill', type=click.Choice(['zero', 'margin']), default=constant.RANDOM_SHIFT_FILL,
              help='Random shift filling value')
@click.option('--random-shift-horizontal', type=int, default=0, help='Random shift input horizontally by x pixels')
@click.option('--random-shift-vertical', type=int, default=0, help='Random shift input vertically by x pixels')
@click.option('--random-scale', type=float, default=0, help='Random scale input data globally by 2^scale')
@click.option('--random-bad-channel', type=float, multiple=True, default=[],
              help='Random (with a probability of 0.5 for each image) assign a pixel as specified value, usually [-1, 0, 1]')
@click.option('--window', type=int, default=1, help='Multi-frame as image channels')
@click.option('--adabn', is_flag=True, help='AdaBN for model adaptation must be used with --minibatch')
@click.option('--num-adabn-epoch', type=int, default=constant.NUM_ADABN_EPOCH)
@click.option('--minibatch', is_flag=True, help='split data into minibatch by subject id')
@click.option('--balance-gesture', type=float, default=0)
@click.option('--amplitude-weighting', is_flag=True)
@click.option('--num-stream', type=int, default=1, help='Number of multi views, only use after multi_stream')
@click.option('--preprocess', callback=lambda ctx, param, value: Preprocess.parse(value))
@click.option('--imu-preprocess', callback=lambda ctx, param, value: Preprocess.parse(value))
@click.option('--batch-size', type=int, default=1000, help='Batch size')
@click.option('--feature-list', cls=PythonLiteralOption,  default='[]', help='Features selected to use')
@click.option('--module', type=click.Choice(['convnet',
                                             'knn',
                                             'svm',
                                             'random-forests',
                                             'lda']), default='convnet')
# Need to add other dataset choices
@click.option('--dataset', type=click.Choice(['ninapro-db1', 'ninapro-db2', 'ninapro-db3',
                                              'ninapro-db4', 'ninapro-db5', 'ninapro-db6',
                                              'ninapro-db7', 'biopatrec-db1', 'biopatrec-db2',
                                              'biopatrec-db3', 'biopatrec-db4']), required=True)
@click.option('--feature-name', type=click.Choice(['featuresigimg_v2',
                                                   'rawsemg_feature_multisource_multistream_sigimgv2',
                                                   'featuresigimg_imuactimg',
                                                   'featuresigimg_imufeature',
                                                   'rawsemg_feature_imu_multisource_multistream_sigimgv1',
                                                   'rawsemg_feature_imu_multisource_multistream_sigimgv2']), required=True)
@click.option('--fusion-type', type=click.Choice(['single', 'multi_no_imu', 'multi_with_imu']), default='single')


@packargs
def exp(args):
    pass


@exp.command()
@click.option('--fold', type=int, required=True, help='Fold number of the crossval experiment')
@click.option('--crossval-type', type=click.Choice(['inter-subject',
                                                    'universal-intra-subject',
                                                    'one-fold-intra-subject',
                                                    'one-fold-intra-subject-adabn',
                                                    'universal-one-fold-intra-subject',
                                                    'universal-one-fold-intra-subject-adabn',
                                                    'temp-one-fold-intra-subject',
                                                    '4-fold-intra-subject',
                                                    '4-fold-inter-subject']), required=True)


@packargs
def crossval(args):
    if args.root:
        if args.log:
            args.log = os.path.join(args.root, args.log)
        if args.snapshot:
            args.snapshot = os.path.join(args.root, args.snapshot)

    if args.gpu_x:
        args.gpu = sum([list(args.gpu) for i in range(args.gpu_x)])
    with Context(args.log, parallel=True):
        logger.info('Args:\n{}', pformat(args))
        for i in range(args.num_epoch):
            path = args.snapshot + '-%04d.params' % (i + 1)
            if os.path.exists(path):
                logger.info('Found snapshot {}, exit', path)
                return
        dataset = Dataset.from_name(args.dataset)
        get_crossval_data = getattr(dataset, 'get_%s_data' % args.crossval_type.replace('-', '_'))
        train, val = get_crossval_data(
            batch_size=args.batch_size,
            fold=args.fold,
            preprocess=args.preprocess,
            imu_preprocess=args.imu_preprocess,
            adabn=args.adabn,
            minibatch=args.minibatch,
            balance_gesture=args.balance_gesture,
            amplitude_weighting=args.amplitude_weighting,
            random_shift_horizontal=args.random_shift_horizontal,
            random_shift_vertical=args.random_shift_vertical,
            random_shift_fill=args.random_shift_fill,
            feature_name=args.feature_name,
            feature_list=args.feature_list,
            window=args.window,
            num_semg_row=args.num_semg_row,
            num_semg_col=args.num_semg_col)
        logger.info('Train samples: {}', train.num_sample)
        logger.info('Val samples: {}', val.num_sample)

        if args.feature_name == 'featuresigimg_v2' or \
                args.feature_name == 'featuresigimg_imuactimg':
            semg_rows = train.shape1
            semg_cols = train.shape2
        elif args.feature_name == 'featuresigimg_imufeature':
            semg_rows = train.semg_shape1
            semg_cols = train.semg_shape2
        elif args.feature_name == 'rawsemg_feature_multisource_multistream_sigimgv2':
            semg_rows = train.shapes1
            semg_cols = train.shapes2
        elif args.feature_name == 'rawsemg_feature_imu_multisource_multistream_sigimgv1':
            semg_rows = train.shapes1
            semg_cols = train.shapes2
        elif args.feature_name == 'rawsemg_feature_imu_multisource_multistream_sigimgv2':
            semg_rows = train.shapes1
            semg_cols = train.shapes2
        else:
            semg_rows = 1
            semg_cols = 1

        if type(train.num_channel) is types.ListType:
            logger.info('Use Multi-stream CNN')
            multi_stream = True
        else:
            logger.info('Use Single-stream CNN')
            multi_stream = False

        mod = Module.parse(
            args.module,
            revgrad=args.revgrad,
            num_revgrad_batch=args.num_revgrad_batch,
            tzeng=args.tzeng,
            num_tzeng_batch=args.num_tzeng_batch,
            num_gesture=train.num_gesture,
            num_subject=train.num_subject,
            subject_loss_weight=args.subject_loss_weight,
            lambda_scale=args.lambda_scale,
            adabn=args.adabn,
            num_adabn_epoch=args.num_adabn_epoch,
            random_scale=args.random_scale,
            multi_stream=multi_stream,
            num_stream=1 if isinstance(train.num_channel, int) else len(train.num_channel),
            for_training=True,
            faug=args.faug,
            faug_classwise=args.faug_classwise,
            num_eval_epoch=args.num_eval_epoch,
            snapshot_period=args.snapshot_period,
            pixel_same_init=args.pixel_same_init,
            symbol_kargs=dict(
                num_semg_row=semg_rows,
                num_semg_col=semg_cols,
                num_filter=args.num_filter,
                num_pixel=args.num_pixel,
                num_feature_block=args.num_feature_block,
                num_gesture_block=args.num_gesture_block,
                num_subject_block=args.num_subject_block,
                num_hidden=args.num_hidden,
                num_bottleneck=args.num_bottleneck,
                dropout=args.dropout,
                num_channel=train.num_channel,
                num_presnet=args.num_presnet,
                presnet_branch=args.presnet_branch,
                drop_presnet=args.drop_presnet,
                bng=args.bng,
                fusion_type=args.fusion_type,
                subject_confusion_loss_weight=args.subject_confusion_loss_weight,
                minibatch=args.minibatch,
                confuse_conv=args.confuse_conv,
                confuse_all=args.confuse_all,
                subject_wd=args.subject_wd,
                drop_branch=args.drop_branch,
                pool=args.pool,
                zscore=args.zscore,
                zscore_bng=args.zscore_bng,
                num_stream=1 if isinstance(train.num_channel, int) else len(train.num_channel),
                drop_conv=args.drop_conv,
                drop_presnet_branch=args.drop_presnet_branch,
                drop_presnet_proj=args.drop_presnet_proj,
                presnet_proj_type=args.presnet_proj_type,
                presnet_promote=args.presnet_promote,
                pixel_reduce_loss_weight=args.pixel_reduce_loss_weight,
                pixel_reduce_bias=args.pixel_reduce_bias,
                pixel_reduce_kernel=args.pixel_reduce_kernel,
                pixel_reduce_stride=args.pixel_reduce_stride,
                pixel_reduce_pad=args.pixel_reduce_pad,
                pixel_reduce_norm=args.pixel_reduce_norm,
                pixel_reduce_reg_out=args.pixel_reduce_reg_out,
                num_pixel_reduce_filter=args.num_pixel_reduce_filter,
                fast_pixel_reduce=args.fast_pixel_reduce,
                drop_pixel=args.drop_pixel,
                num_conv=args.num_conv,
                presnet_dense=args.presnet_dense,
                conv_shortcut=args.conv_shortcut
            ),
            context=[mx.gpu(i) for i in args.gpu]
        )
        mod.fit(
            train_data=train,
            eval_data=val,
            num_epoch=args.num_epoch,
            num_train=train.num_sample,
            batch_size=args.batch_size,
            lr_step=args.lr_step,
            lr_factor=args.lr_factor,
            lr=args.lr,
            wd=args.wd,
            gamma=args.gamma,
            snapshot=args.snapshot,
            params=args.params,
            ignore_params=args.ignore_params,
            fix_params=args.fix_params,
            decay_all=args.decay_all)


if __name__ == '__main__':
    cli(obj=Bunch())








