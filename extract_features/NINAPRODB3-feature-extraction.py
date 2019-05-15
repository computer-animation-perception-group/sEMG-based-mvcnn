import os
import scipy.io as sio
import numpy as np
import emg_features
#from activity_image import get_signal_img
from itertools import product
from collections import namedtuple
from joblib import Parallel, delayed
#import cv2
#from ..utils import butter_lowpass_filter as lowpass


subjects = list(range(5, 6))
gestures=list(range(50))
trials = list(range(6))
#input_path = '/home/weiwentao/public/duyu/misc/ninapro-db1'
#output_path = '/home/weiwentao/public/semg/ninapro-feature/ninapro-db1-var-raw'

home = os.path.expanduser('~')

input_path = os.path.join(home, '10.214.150.99/semg/dqf/data/ninapro-db3/semg/data')


filtering_type = 'lowpass'
framerate = 2000

window_length_ms = 200
window_stride_ms = 10

window = window_length_ms*100/1000
stride = window_stride_ms*100/1000

output_path = os.path.join(home, '10.214.150.99/semg/dqf/data/ninapro-db3/feature/ninapro-db3-var-raw-downsample20-prepro-%s-win-%d-stride-%d' % (filtering_type, window, stride))

Combo = namedtuple('Combo', ['subject', 'gesture', 'trial'], verbose=False)


#feature_list = ['dwt','dwpt','dwpt_mean','dwpt_sd','mav','mavslpphinyomark','wl','sscbestninapro1','rms','hemg20','mdwtdb1ninapro']


#feature_list = ['aac','afb9','apen','dwpt_energy','dwpt_kurtosis','dwpt_m2','dwpt_m3','dwpt_m4','dwpt_skewness',
#                'fr', 'hemg15', 'hos', 'mavtm3', 'mavtm4', 'mavtm5', 'mdf_MEDIAN_POWER', 'mhw_energy', 'mnfhht',
#                'mnp', 'mtw_energy', 'myop', 'pkf', 'smn1', 'smn2', 'smn3', 'ssc',
#                'ssi', 'ttp', 'var', 'vcf', 'wfe', 'wl_dasdv', 'wte', 'arr29', 'hht58', dwt_energy']

#feature_list = ['mnp', 'mtw_energy', 'myop', 'pkf', 'smn1', 'smn2', 'smn3', 'ssc']

#feature_list = ['iemg', 'rms', 'mdwt', 'hemg15', 'tdd_cor']
#feature_list = [ 'hht58', 'arr29']
#feature_list = [ 'ssc', 'zc', 'var']
#feature_list = [ 'mnf', 'cc']
#feature_list = ['dwpt', 'dwt', 'mav','wl','wamp','mavslpframewise','arc','mnf_MEDIAN_POWER','psr', 'tdd_cor']
feature_list = ['arc']


def get_combos(*args):
   for arg in args:
       if isinstance(arg, tuple):
           arg = [arg]
       for a in arg:
           yield Combo(*a)



#the following functions can be loaded from ..utils


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
        (window-stride)* data.shape[1]
    )

def windowed_view(arr, window, overlap):
    from numpy.lib.stride_tricks import as_strided
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                  window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                   arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)


#def dft(data):
#     f = np.fft.fft2(data)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20*np.log(np.abs(fshift))
#     return magnitude_spectrum
#
##the following functions can be loaded from ..
#
#def dft_dy(data):
#    data = data.T
#    n = data.shape[-1]
#    window = np.hanning(n)
#    windowed = data * window
#    spectrum = np.fft.fft(windowed)
#    return np.abs(spectrum)


def feature_map(x):

    res = []
    for i in range(x.shape[0]):
        single_channel = []
        for j in range(len(feature_list)):
            func = 'emg_features.emg_'+feature_list[j]
            single_channel.append(eval(str(func))(x[i,:]))
        # print single_channel[0].shape
        single_channel = np.hstack(single_channel)
        res.append(single_channel)
    res =np.vstack(res)
    return res

def extract_emg_feature(x, feature_name):
    res = []
    for i in range(x.shape[0]):
        func = 'emg_features.emg_'+feature_name
        res.append(eval(str(func))(x[i,:]))
    res =np.vstack(res)
    return res

def emg_feature_extraction_parallel(out_dir, combo, data, feature_name):
     feature = [np.transpose(extract_emg_feature(seg.T, feature_name)) for seg in data]
     feature = np.array(feature)
     out_path = os.path.join(
                              out_dir,
                              '{0.subject:03d}_{0.gesture:03d}_{0.trial:03d}_{1}.mat').format(combo, feature_name)
     sio.savemat(out_path, {'data': feature, 'label': combo.gesture, 'subject': combo.subject, 'trial':combo.trial})
     print ("Subject %d Gesture %d Trial %d %s saved!" % (combo.subject, combo.gesture, combo.trial, feature_name))


def creating_dir_parallel(input_path, output_path, combo):


      out_dir = os.path.join(
                output_path,
                '{c.subject:03d}',
                '{c.gesture:03d}').format(c=combo)

      if os.path.isdir(out_dir) is False:
             os.makedirs(out_dir)

      print ("Subject %d Gesture %d DIR maded!" % (combo.subject, combo.gesture))



def emg_feature_extraction_parallel_2(input_path, output_path, combo, feature_list):
      in_path = os.path.join(
                input_path,
                '{c.subject:03d}',
                '{c.gesture:03d}',
                '{c.subject:03d}_{c.gesture:03d}_{c.trial:03d}.mat').format(c=combo)

      out_dir = os.path.join(
                output_path,
                '{c.subject:03d}',
                '{c.gesture:03d}').format(c=combo)

      if os.path.isdir(out_dir) is False:
             os.makedirs(out_dir)


      data = sio.loadmat(in_path)['data'].astype(np.float32)

      print ("Subject %d Gesture %d Trial %d data loaded!" % (combo.subject, combo.gesture, combo.trial))

      data = np.abs(data)

      if filtering_type is 'lowpass':
#             data = np.transpose([lowpass(ch, 1, framerate, 1, zero_phase=True) for ch in data.T])
             data = np.transpose([butter_lowpass_filter(ch, 1, framerate, 1, zero_phase=True) for ch in data.T])
             print ("Subject %d Gesture %d Trial %d bandpass filtering finished!" % (combo.subject, combo.gesture, combo.trial))
      else:
             pass

      data = downsample(data, step=20)

      chnum = data.shape[1];
      data = get_segments(data, window, stride)
      data = data.reshape(-1, window, chnum)

      for feature_name in feature_list:
          out_path = os.path.join(
            out_dir,
            '{0.subject:03d}_{0.gesture:03d}_{0.trial:03d}_{1}.mat').format(combo, feature_name)
          if not os.path.exists(out_path):
            feature = [np.transpose(extract_emg_feature(seg.T, feature_name)) for seg in data]
            feature = np.array(feature)
            sio.savemat(out_path, {'data': feature, 'label': combo.gesture, 'subject': combo.subject, 'trial':combo.trial})
            print ("Subject %d Gesture %d Trial %d %s saved!" % (combo.subject, combo.gesture, combo.trial, feature_name))


if __name__ == '__main__':

    print ("NinaPro feature map generation, use window = %d frames, stride = %d frames" % (window, stride))

    combos = get_combos(product(subjects, gestures, trials))

    combos = list(combos)


#    # for test only
#    pre_data = []
#    feature_dim = 0
#    for i in range(len(feature_list)):
#        input_dir = os.path.join(output_path,
#                            '000',
#                            '000',
#                            '000_000_000_{0}.mat'
#                           ).format(feature_list[i])
#        mat = sio.loadmat(input_dir)
#        data = mat['data'].astype(np.float32)
#        feature_dim = feature_dim + data.shape[1]
#        pre_data.append(data)
#    pre_data = np.concatenate(pre_data, axis=1)
#    feature_dim = pre_data.shape[1]

    #Parallel(n_jobs=8)(delayed(creating_dir_parallel)(input_path, output_path, combo) for combo in combos)

    Parallel(n_jobs=8)(delayed(emg_feature_extraction_parallel_2)(input_path, output_path, combo, feature_list) for combo in combos)


#    for combo in combos:
#        in_path = os.path.join(
#                input_path, 'data',
#                '{c.subject:03d}',
#                '{c.gesture:03d}',
#                '{c.subject:03d}_{c.gesture:03d}_{c.trial:03d}.mat').format(c=combo)
#
#        out_dir = os.path.join(
#                output_path,
#                '{c.subject:03d}',
#                '{c.gesture:03d}').format(c=combo)
#
#        if os.path.isdir(out_dir) is False:
#             os.makedirs(out_dir)
#
#
#        data = sio.loadmat(in_path)['data'].astype(np.float32)
#
#        print ("Subject %d Gesture %d Trial %d data loaded!" % (combo.subject, combo.gesture, combo.trial))
#
#        if filtering_type is 'lowpass':
##             data = np.transpose([lowpass(ch, 1, framerate, 1, zero_phase=True) for ch in data.T])
#             data = np.transpose([butter_lowpass_filter(ch, 1, framerate, 1, zero_phase=True) for ch in data.T])
#             print ("Subject %d Gesture %d Trial %d bandpass filtering finished!" % (combo.subject, combo.gesture, combo.trial))
#        else:
#             pass
#
#
#
#        chnum = data.shape[1];
#        data = get_segments(data, window, stride)
#        data = data.reshape(-1, window, chnum)
#
#        Parallel(n_jobs=4)(delayed(emg_feature_extraction_parallel)(out_dir, combo, data, feature_list[i]) for i in range(len(feature_list)))
#
##        for i in range(len(feature_list)):
##            feature = [np.transpose(extract_emg_feature(seg.T, feature_list[i])) for seg in data]
##            feature = np.array(feature)
##            out_path = os.path.join(
##                                    out_dir,
##                                    '{0.subject:03d}_{0.gesture:03d}_{0.trial:03d}_{1}.mat').format(combo, feature_list[i])
##            sio.savemat(out_path, {'data': feature, 'label': combo.gesture, 'subject': combo.subject, 'trial':combo.trial})
##            print ("Subject %d Gesture %d Trial %d %s saved!" % (combo.subject, combo.gesture, combo.trial, feature_list[i]))












