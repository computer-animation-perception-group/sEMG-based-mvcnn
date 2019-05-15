import os
import scipy.io as sio
import numpy as np
import emg_features
from itertools import product
from collections import namedtuple
from joblib import Parallel, delayed
import argparse
from tqdm import tqdm
from logbook import Logger, StderrHandler, NullHandler, FileHandler
import warnings
#import threading
#import multiprocessing


Combo = namedtuple('Combo', ['subject', 'gesture', 'trial'], verbose=False)
log = Logger('Extract features')

config = {}
config['ninapro-db1'] = dict(
    subjects=list(range(0, 27)),
    gestures=list(range(1, 53)),
    trials=list(range(10)),
    framerate=100
)
config['ninapro-db2'] = dict(
    subjects=list(range(40)),
    gestures=list(range(50)),
    trials=list(range(6)),
    framerate=2000
)
config['ninapro-db3'] = dict(
    subjects=list(range(2, 11)),
    gestures=list(range(50)),
    trials=list(range(10)),
    framerate=2000
)
config['ninapro-db4'] = dict(
    subjects=list(range(0, 27)),
    gestures=list(range(1, 53)),
    trials=list(range(6)),
    framerate=2000
)
config['ninapro-db5'] = dict(
    subjects=list(range(10)),
    gestures=list(range(53)),
    trials=list(range(6)),
    framerate=200
)
config['ninapro-db6'] = dict(
    subjects=[0, 2, 3, 4, 5, 6, 7, 9],
    gestures=list(range(7)),
    trials=list(range(120)),
    framerate=2000
)
config['ninapro-db7'] = dict(
    subjects=list(range(22)),
    gestures=list(range(41)),
    trials=list(range(6)),
    framerate=2000
)
config['biopatrec-db1'] = dict(
    subjects=list(range(20)),
    gestures=list(range(10)),
    trials=list(range(3)),
    framerate=2000
)
config['biopatrec-db2'] = dict(
    subjects=list(range(17)),
    gestures=list(range(26)),
    trials=list(range(3)),
    framerate=2000
)
config['biopatrec-db3'] = dict(
    subjects=list(range(8)),
    gestures=list(range(10)),
    trials=list(range(3)),
    framerate=2000
)
config['biopatrec-db4'] = dict(
    subjects=list(range(8)),
    gestures=list(range(8)),
    trials=list(range(3)),
    framerate=2000
)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        raise RuntimeError('{} is not a dir'.format(path))
    return path


def main():
    parser = argparse.ArgumentParser(description='Extract features')
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help='Raw data input dir'
    )
    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help='Output dir'
    )
    parser.add_argument(
        '--filter',
        default='lowpass',
        help='Filtering Type'
    )
    parser.add_argument(
        '--window',
        type=int,
        required=True,
        help='Window length'
    )
    parser.add_argument(
        '--stride',
        type=int,
        required=True,
        help='Stride length'
    )
    parser.add_argument(
        '-f',
        '--featurelist',
        nargs='+',
        help='Features to extact',
        required=True
    )
    parser.add_argument(
        '--downsample',
        type=int,
        default=1,
        help='Downsample step, default takes no downsample'
    )
    parser.add_argument(
        '--log',
        default='info',
        choices=['debug', 'warning', 'info', 'error'],
        help='Logging level, default info'
    )
    parser.add_argument(
        '--dataset',
        choices=['ninapro-db1', 'ninapro-db2', 'ninapro-db3', 'ninapro-db4',
                 'ninapro-db5', 'ninapro-db6', 'ninapro-db7', 'biopatrec-db1',
                 'biopatrec-db2', 'biopatrec-db3', 'biopatrec-db4'],
        help='Dataset choices',
        required=True
    )

    args = parser.parse_args()

    with NullHandler().applicationbound():
        with StderrHandler(level=args.log.upper()).applicationbound():
            with FileHandler(
                os.path.join(ensure_dir(args.output), 'log'),
                level=args.log.upper(),
                bubble=True
            ).applicationbound():
                try:
                    return run(args)
                except:
                    log.exception('Failed')


def get_combos(*args):
    for arg in args:
        if isinstance(arg, tuple):
            arg = [arg]
        for a in arg:
            yield Combo(*a)


def downsample(data, step):
    return data[::step].copy()


def butter_lowpass_filter(data, cut, fs, order, zero_phase=False):
    from scipy.signal import butter, lfilter, filtfilt

    nyq = 0.5 * fs
    cut = cut / nyq

    b, a = butter(order, cut, btype='low')
    y = (filtfilt if zero_phase else lfilter)(b, a, data)
    return y


def get_segments(data, window, stride):
    return windowed_view(
        data.flat,
        window * data.shape[1],
        (window - stride) * data.shape[1])


def windowed_view(arr, window, overlap):
    from numpy.lib.stride_tricks import as_strided
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                  window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                  arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)


def extract_emg_feature(x, feature_name):
    if feature_name == 'tsd_v1':
        res = []
        func = 'emg_features.emg_tdd_cor'
        for i in range(x.shape[0]):
            res.append(eval(str(func))(x[i,:]))
        for i in range(x.shape[0]):
            for j in range(i+1,x.shape[0]):
                res.append(eval(str(func))(x[i,:]-x[j,:]))
        return np.vstack(res)
    else:
        res = []
        for i in range(x.shape[0]):
            func = 'emg_features.emg_'+feature_name
            res.append(eval(str(func))(x[i,:]))
        res =np.vstack(res)
        return res


def extract(input_path, output_path, combo, feature_list, **kargs):
    dataset = kargs.pop('dataset', 'ninapro-db1')
    filter_type = kargs.pop('filter', 'lowpass')
    downsample = kargs.pop('downsample', 1)
    window = kargs.pop('window')
    stride = kargs.pop('stride')
    framerate = kargs.pop('framerate')

    source_path = os.path.join(
        input_path,
        '{c.subject:03d}',
        '{c.gesture:03d}',
        '{c.subject:03d}_{c.gesture:03d}_{c.trial:03d}.mat').format(c=combo)

    if os.path.exists(source_path):
        data = sio.loadmat(source_path)['data'].astype(np.float32)
    else:
        log.debug('{} not exists'.format(source_path))
        return

    log.debug('Subject %d Gesture %d Trial %d' %
              (combo.subject, combo.gesture, combo.trial))

    output_path = os.path.join(output_path, dataset)

    if filter_type is 'lowpass':
        data = np.transpose([butter_lowpass_filter(ch, 1, framerate, 1, zero_phase=True)
                             for ch in data.T])
        output_path = output_path + '-lowpass'
        log.debug('Bandpass filering finish')
    else:
        pass

    if downsample > 1:
        data = downsample(data, step=downsample)
        output_path = output_path + '-downsample%d' % (downsample)

    chnum = data.shape[1]
    data = get_segments(data, window, stride)
    data = data.reshape(-1, window, chnum)

    output_path = '{0}-win-{1}-stride-{2}'.format(output_path,
                                                   window,
                                                   stride)

    target_path = os.path.join(
        output_path,
        '{c.subject:03d}',
        '{c.gesture:03d}').format(c=combo)

    feature_list = tqdm(feature_list, leave=False)
    for feature_name in feature_list:
        feature_list.set_description("Extracting feature %s ..." % feature_name)
        feature = [np.transpose(extract_emg_feature(seg.T, feature_name)) for seg in data]
        feature = np.array(feature)
        out_path = os.path.join(
            ensure_dir(target_path),
            '{0.subject:03d}_{0.gesture:03d}_{0.trial:03d}_{1}.mat').format(combo, feature_name)
        sio.savemat(out_path, {
            'data': feature,
            'label': combo.gesture,
            'subject': combo.subject,
            'trial': combo.trial
        })

def run(args):

    local_config = config[args.dataset]

    log.info('Extract feature of %s...' % args.dataset)

    combos = get_combos(product(local_config['subjects'],
                                local_config['gestures'],
                                local_config['trials']))
    combos = list(combos)

    framerate = local_config['framerate']
    window = args.window
    stride = args.stride

    res = dict(
        dataset = args.dataset,
        filter_type = args.filter,
        downsample = args.downsample,
        window = window,
        stride = stride,
        framerate = framerate
    )
    return args.input, args.output, combos, args.featurelist, res


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    input_path, output_path, combos, featurelist, para = main()

    #for combo in tqdm(combos,
    #                  desc=para['dataset'],
    #                  leave=False):
    #    extract(input_path, output_path, combo, featurelist, **para)
    Parallel(n_jobs=8)(delayed(extract)(input_path,
                                        output_path,
                                        combo, featurelist, **para)
                       for combo in tqdm(combos, leave=False))


