from __future__ import division
import os
from nose.tools import assert_equal
from functools import partial
from logbook import Logger
import numpy as np
import scipy.io as sio
#import scipy.stats as sstat
from itertools import product
from collections import OrderedDict
#from lru import LRU
from . import Dataset as Base, PREPROCESS_KARGS
from .. import Combo, Trial
from ... import utils, constant
from sklearn.model_selection import KFold
from SINGLESRC_featuresigimgv2_iter import FeatureSigImgData_v2
from MULTISRC_rawemg_feature_sigimgv2_iter import Feature_RawSEMG_Multistream_SigImg_Iter
from SINGLESRC_featuresigimg_imuv1_iter import FeatureSigImg_ImuData_v1
from SINGLESRC_featuresigimg_imuv2_iter import FeatureSigImg_ImuData_v2
from MULTISRC_rawemg_feature_imu_multistream_iter import Feature_RawSEMG_IMU_Multistream_SigImg_Iter_v1
from MULTISRC_rawemg_feature_imu_multistream_iter_v2 import Feature_RawSEMG_IMU_Multistream_SigImg_Iter_v2


logger =  Logger(__name__)

# ninapro 3
WINDOW = 20
STRIDE = 1

class Dataset(Base):
    name = 'ninapro-base'

    num_semg_row = constant.NUM_SEMG_ROW
    num_semg_col = constant.NUM_SEMG_COL

    feature_extraction_winlength = constant.FEATURE_EXTRACTION_WIN_LEN
    feature_extraction_winstride = constant.FEATURE_EXTRACTION_WIN_STRIDE

    feature_names = constant.FEATURE_LIST

    subjects = list(range(27)) # Need to be overriding, it's for db1
    gestures = list(range(1, 53))
    trials = list(range(10))

    def get_trial_func(self, *args, **kargs):
        return GetTrial(*args, **kargs)

    def get_dataiter(self, get_trial, combos, feature_name,
                     window=1,
                     adabn=False,
                     mean=None,
                     dense_window=False,
                     scale=None,
                     **kargs):
        logger.info("Use %s, semg row = %d, semg col = %d, window=%d" % (feature_name,
                                                                         self.num_semg_row,
                                                                         self.num_semg_col,
                                                                         window))
        if feature_name == 'featuresigimg_v2' or \
                feature_name == 'featuresigimg_imuactimg' or \
                feature_name == 'featuresigimg_imufeature':
            self.semg_root = None

        if feature_name == 'featuresigimg_v2' or \
                feature_name == 'rawsemg_feature_multisource_multistream_sigimgv2':
            self.imu_root = None

        def data_scale(data):
            if mean is not None:
                data = data - mean
            if scale is not None:
                data = data * scale
            return data

        combos = list(combos)

        data = []
        feature = []
        imu = []
        gesture = []
        subject = []
        segment = []

        if self.semg_root is None:
            # just feature and imu
            for combo in combos:
                trial = get_trial(self.semg_root, self.feature_root, self.imu_root, combo=combo)
                feature.append(data_scale(trial.data[0]))
                if self.imu_root is not None:
                    imu.append(data_scale(trial.data[1])) # trial.data: (feature:..),(imu:..)
                gesture.append(trial.gesture)
                subject.append(trial.subject)
                segment.append(np.repeat(len(segment), len(feature[-1])))
                logger.debug('MAT loaded')
                data = feature
        else:
            for combo in combos:
                trial = get_trial(self.semg_root, self.feature_root, self.imu_root, combo=combo)
                data.append(data_scale(trial.data[0]))
                feature.append(data_scale(trial.data[1]))
                if self.imu_root is not None:
                    imu.append(data_scale(trial.data[2]))
                gesture.append(trial.gesture)
                subject.append(trial.subject)
                segment.append(np.repeat(len(segment), len(data[-1])))
                logger.debug('MAT loaded')

        if not data:
            logger.warn('Empty data')
            return

        index = []
        n = 0
        for seg in data:
            index.append(np.arange(n, n + len(seg) - window + 1))
            n += len(seg)
        index = np.hstack(index)
        logger.debug('Index made')
        logger.debug('Segments: {}', len(data))

        logger.debug('First segment data shape: {}', data[0].shape)
        data = np.vstack(data)
        if len(data.shape) == 2:
            print('Using 2D data')
            data = data.reshape(data.shape[0], 1, 1, -1)
        else:
            data = data.reshape(data.shape[0], 1, -1, self.num_semg_row * self.num_semg_col)
        logger.debug('Reshaped data shape: {}', data.shape)
        print('data shape: ', data.shape)

        logger.debug('First segment feature shape: {}', feature[0].shape)
        feature = np.vstack(feature)
        if len(feature.shape) == 2:
            print('Using 2D feature')
            feature = feature.reshape(feature.shape[0], 1, 1, -1)
        else:
            feature = feature.reshape(feature.shape[0], 1, -1, self.num_semg_row * self.num_semg_col)
        logger.debug('Reshaped feature shape: {}', feature.shape)

        if self.imu_root is not None:
            logger.debug('First segment imu shape: {}', imu[0].shape)
            imu = np.vstack(imu)
            if len(imu.shape) == 2:
                print('Using 2D imu')
                imu = imu.reshape(imu.shape[0], 1, 1, -1)
            else:
                imu = imu.reshape(imu.shape[0], 1, -1, self.num_imu_row * self.num_imu_col)
            logger.debug('Reshaped imu shape: {}', imu.shape)
            print('imu shape:', imu.shape)

        logger.debug('Data and feature stacked')

        gesture = get_index(np.hstack(gesture))
        subject = get_index(np.hstack(subject))
        segment = np.hstack(segment)

        label = []

        label.append(('gesture_softmax_label', gesture))

        logger.debug('Make data iter')

        if feature_name == 'featuresigimg_v2':
            # Single view
            assert self.semg_root is None and self.imu_root is None
            return FeatureSigImgData_v2(
                data=OrderedDict([('data', data)]),
                label=OrderedDict(label),
                gesture=gesture.copy(),
                subject=subject.copy(),
                segment=segment.copy(),
                index=index,
                adabn=adabn,
                window=window,
                num_gesture=gesture.max() + 1,
                num_subject=subject.max() + 1,
                **kargs)
        elif feature_name == 'rawsemg_feature_multisource_multistream_sigimgv2':
            # Multi view
            #assert self.semg_root is not None and self.imu_root is None
            return Feature_RawSEMG_Multistream_SigImg_Iter(
                data=OrderedDict([('feature', feature)]),
                label=OrderedDict(label),
                gesture=gesture.copy(),
                subject=subject.copy(),
                segment=segment.copy(),
                index=index,
                adabn=adabn,
                window=window,
                num_gesture=gesture.max() + 1,
                num_subject=subject.max() + 1,
                **kargs)
        elif feature_name == 'featuresigimg_imuactimg':
            # Single view with activity image of imu
            assert self.imu_root is not None and self.semg_root is None
            #data = np.concatenate((data, imu), axis=2)
            return FeatureSigImg_ImuData_v1(
                data=OrderedDict([('data', data), ('imu', imu)]),
                label=OrderedDict(label),
                gesture=gesture.copy(),
                subject=subject.copy(),
                segment=segment.copy(),
                index=index,
                adabn=adabn,
                window=window,
                num_gesture=gesture.max() + 1,
                num_subject=subject.max() + 1,
                **kargs)
        elif feature_name == 'featuresigimg_imufeature':
            # Single view with feature image of imu
            assert self.imu_root is not None and self.semg_root is None
            return FeatureSigImg_ImuData_v2(
                data=OrderedDict([('data', data), ('imu', imu)]),
                label=OrderedDict(label),
                gesture=gesture.copy(),
                subject=subject.copy(),
                segment=segment.copy(),
                index=index,
                adabn=adabn,
                window=window,
                num_gesture=gesture.max() + 1,
                num_subject=subject.max() + 1,
                **kargs)
        elif feature_name == 'rawsemg_feature_imu_multisource_multistream_sigimgv1':
            # Multi view with activity image of imu
            assert self.imu_root is not None
            return Feature_RawSEMG_IMU_Multistream_SigImg_Iter_v1(
                data=OrderedDict([('feature', feature), ('imu', imu)]),
                label=OrderedDict(label),
                gesture=gesture.copy(),
                subject=subject.copy(),
                segment=segment.copy(),
                index=index,
                adabn=adabn,
                window=window,
                num_gesture=gesture.max() + 1,
                num_subject=subject.max() + 1,
                **kargs)
        elif feature_name == 'rawsemg_feature_imu_multisource_multistream_sigimgv2':
            # Multi view with feature image of imu
            assert self.imu_root is not None and self.semg_root is not None
            return Feature_RawSEMG_IMU_Multistream_SigImg_Iter_v2(
                data=OrderedDict([('semg', data), ('feature', feature), ('imu', imu)]),
                label=OrderedDict(label),
                gesture=gesture.copy(),
                subject=subject.copy(),
                segment=segment.copy(),
                index=index,
                adabn=adabn,
                window=window,
                num_gesture=gesture.max() + 1,
                num_subject=subject.max() + 1,
                **kargs)

    def get_one_fold_intra_subjects_trials(self):
        return [0, 2, 3, 5, 7, 8, 9], [1, 4, 6]

    def get_one_fold_intra_subject_adabn_trials(self):
        return self.trials[::2], self.trials[1::2]

    #def get_k_fold_inter_subject(self, k):
    #    folds = []
    #    step = int(math.ceil(len(self.subjects) / float(k)))
    #    for i in range(0, len(self.subjects), step):
    #        folds.append(self.subjects[i: i + step])
    #    return folds
    def get_k_fold_inter_subject(self, k):
        kf = KFold(n_splits=k)
        folds = []
        for _, val_subjects in kf.split(self.subjects):
            temp = []
            for i in val_subjects:
                temp.append(self.subjects[i])
            folds.append(temp)
        #step = int(math.ceil(len(self.subjects) / float(k)))
        #for i in range(0, len(self.subjects), step):
        #    folds.append(self.subjects[i: i + step])
        return folds

    def get_4_fold_inter_subject_data(self, fold, batch_size, preprocess, imu_preprocess,
                                      adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, feature_list=[], **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        folds = self.get_k_fold_inter_subject(4)
        val_subjects = folds[fold]
        train_subjects = [i for i in self.subjects if i not in val_subjects]
        train = load(
            combos=self.get_combos(product(train_subjects, self.gestures, self.trials)),
            adabn=adabn,
            mini_batch_size=batch_size // (4 if minibatch else 1),
            shuffle=True,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product(val_subjects, self.gestures, self.trials)),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val

    def get_inter_subject_data(self, fold, batch_size, preprocess, imu_preprocess,
                               adabn, minibatch, feature_name, window, num_semg_row, num_semg_col,
                               feature_list=[], **kwargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train = load(
            combos=self.get_combos(product([i for i in self.subjects if i != subject],
                                           self.gestures, self.trials)),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject - 1 if minibatch else 1),
            shuffle=True,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product([subject], self.gestures, self.trials)),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val

    def get_universal_intra_subject_data(self, fold, batch_size, preprocess, imu_preprocess,
                                         adabn, minibatch, feature_name, window, num_semg_row, num_semg_col,
                                         feature_list=[], **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        trial = self.trials[fold]
        train = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in self.trials if i != trial])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combo=self.get_combos(product(self.subjects, self.gestures, [trial])),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val

    def get_one_fold_intra_subject_adabn_data(self, fold, batch_size, preprocess, imu_preprocess,
                                              adabn, minibatch, feature_name, window, num_semg_row, num_semg_col,
                                              feature_list=[], **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train_trials, val_trials = self.get_one_fold_intra_subject_adabn_trials()
        train = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val


    def get_one_fold_intra_subject_data(self, fold, batch_size, preprocess, imu_preprocess,
                                        adabn, minibatch, feature_name, window, num_semg_row, num_semg_col,
                                        feature_list=[], **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        train = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val

    def get_universal_one_fold_intra_subject_data(self, fold, batch_size, preprocess, imu_preprocess,
                                                  adabn, minibatch, feature_name, window, num_semg_row, num_semg_col,
                                                  feature_list=[], **kargs):
        assert_equal(fold, 0)
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        train = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (10 if minibatch else 1),
            shuffle=True,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val

    def get_universal_one_fold_intra_subject_adabn_data(self, fold, batch_size, preprocess, imu_preprocess,
                                                        adabn, minibatch, feature_name, window, num_semg_row, num_semg_col,
                                                        feature_list=[], **kargs):
        assert_equal(fold, 0)
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        train_trials, val_trials = self.get_one_fold_intra_subject_adabn_trials()
        train = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (10 if minibatch else 1),
            shuffle=True,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val

    def get_one_fold_intra_subject_val(self, fold, batch_size, feature_name, window,
                                       num_semg_row, num_semg_col,
                                       preprocess=None, imu_preprocess=None, feature_list=[], **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        _, val_trials = self.get_one_fold_intra_subject_trials()
        val = load(combos=self.get_combos(product([subject], self.gestures,
                                                  [i for i in val_trials])),
                   shuffle=False,
                   feature_name=feature_name,
                   window=window,
                   num_semg_row=num_semg_row,
                   num_semg_col=num_semg_col)
        return val

    def get_one_fold_intra_subject_adabn_val(self, fold, batch_size, feature_name, window,
                                             num_semg_row, num_semg_col,
                                             preprocess=None, imu_preprocess=None, feature_list=[], **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        _, val_trials = self.get_one_fold_intra_subject_adabn_trials()
        val = load(combos=self.get_combos(product([subject], self.gestures,
                                                  [i for i in val_trials])),
                   shuffle=False,
                   feature_name=feature_name,
                   window=window,
                   num_semg_row=num_semg_row,
                   num_semg_col=num_semg_col)
        return val

    def get_4_fold_inter_subject_val(self, fold, batch_size, feature_name, window,
                                     num_semg_row, num_semg_col,
                                     preprocess=None, imu_preprocess=None, feature_list=[], **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        folds = self.get_k_fold_inter_subject(4)
        val_subjects = folds[fold]
        val = load(
            combos=self.get_combos(product(val_subjects, self.gestures, self.trials)),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col
        )
        return val

    def get_inter_subject_val(self, fold, batch_size, feature_name, window,
                              num_semg_row, num_semg_col,
                              preprocess=None, imu_preprocess=None, feature_list=[], **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        val = load(
            combos=self.get_combos(product([subject], self.gestures, self.trials)),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return val

    def get_4_fold_intra_subject_val(self, fold, batch_size, feature_name, window,
                                     num_semg_row, num_semg_col,
                                     preprocess=None, imu_preprocess=None, feature_list=[], **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        folds = self.get_k_fold_inter_subject(4)
        subjects = folds[fold]
        _, val_trials = self.get_one_fold_intra_subject_trials()
        val = load(
            combos=self.get_combos(product(subjects, self.gestures, val_trials)),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col
        )
        return val

    def get_4_fold_intra_subject_data(self, fold, batch_size, feature_name, window,
                                      adabn, minibatch, num_semg_row, num_semg_col,
                                      preprocess=None, imu_preprocess=None, feature_list=[], **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        folds = self.get_k_fold_inter_subject(4)
        subjects = folds[fold]
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()

        train = load(
            combos=self.get_combos(product(subjects, self.gestures, train_trials)),
            adabn=adabn,
            mini_batch_size=batch_size // (10 if minibatch else 1),
            shuffle=True,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product(subjects, self.gestures, val_trials)),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val

    def get_one_fold_intra_subject_caputo_data(self, fold, batch_size, preprocess, imu_preprocess,
                                               adabn, minibatch, feature_name, window, num_semg_row, num_semg_col,
                                               feature_list=[], **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train_trials, val_trials = self.get_one_fold_intra_subject_caputo_trials()
        train = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
            shuffle=True,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product([subject], self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val

    def get_universal_one_fold_intra_subject_caputo_data(self, fold, batch_size, preprocess, imu_preprocess,
                                                         adabn, minibatch, feature_name, window, num_semg_row, num_semg_col,
                                                         feature_list=[], **kargs):
        assert_equal(fold, 0)
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        train_trials, val_trials = self.get_one_fold_intra_subject_caputo_trials()
        train = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (10 if minibatch else 1),
            shuffle=True,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        val = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in val_trials])),
            shuffle=False,
            feature_name=feature_name,
            window=window,
            num_semg_row=num_semg_row,
            num_semg_col=num_semg_col)
        return train, val

    def get_one_fold_intra_subject_caputo_val(self, fold, batch_size, feature_name, window,
                                              num_semg_row, num_semg_col,
                                              preprocess=None, imu_preprocess=None, feature_list=[], **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names if len(feature_list)==0 else feature_list)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        subject = self.subjects[fold]
        _, val_trials = self.get_one_fold_intra_subject_caputo_trials()
        val = load(combos=self.get_combos(product([subject], self.gestures,
                                                  [i for i in val_trials])),
                   shuffle=False,
                   feature_name=feature_name,
                   window=window,
                   num_semg_row=num_semg_row,
                   num_semg_col=num_semg_col)
        return val


def get_index(arr):
    '''Convert label to 0 based index'''
    b = list(set(arr))
    return np.array([x if x < 0 else b.index(x) for x in arr.ravel()]).reshape(arr.shape)


class GetTrial(object):
    def __init__(self, gestures, trials, preprocess=None, imu_preprocess=None, feature_names=None):
        self.preprocess = preprocess
        self.imu_preprocess = imu_preprocess
        self.memo = {}
        self.gesture_and_trials =  list(product(gestures, trials))
        self.feature_names = feature_names

    def get_path(self, root, combo):
        return os.path.join(
            root,
            '{c.subject:03d}',
            '{c.gesture:03d}',
            '{c.subject:03d}_{c.gesture:03d}_{c.trial:03d}.mat').format(c=combo)

    def get_feature_path(self, root, combo):
        return os.path.join(
            root,
            '{c.subject:03d}',
            '{c.gesture:03d}',
            '{c.subject:03d}_{c.gesture:03d}_{c.trial:03d}').format(c=combo)

    def get_imu_path(self, root, combo):
        return os.path.join(
            root,
            '{c.subject:03d}',
            '{c.gesture:03d}',
            '{c.subject:03d}_{c.gesture:03d}_{c.trial:03d}').format(c=combo)

    def __call__(self, semg_root, feature_root, imu_root, combo):
        assert (feature_root is not None)

        if semg_root is not None:
            path = self.get_path(semg_root, combo)
            if path not in self.memo:
                logger.debug('Load subject semg data {}', combo.subject)
                paths = [self.get_path(semg_root, Combo(combo.subject, gesture, trial))
                         for gesture, trial in self.gesture_and_trials]
                self.memo.update({path: semg_data for path, semg_data in
                                  zip(paths, _get_data(paths, self.preprocess))})
            semg_data = self.memo[path]
            semg_data = semg_data.copy()

        if feature_root is not None:
            path = self.get_feature_path(feature_root, combo)
            if path not in self.memo:
                logger.debug('Load subject feature data {}', combo.subject)
                paths = [self.get_feature_path(feature_root, Combo(combo.subject, gesture, trial))
                         for gesture, trial in self.gesture_and_trials]
                self.memo.update({path: semg_feature for path, semg_feature in
                                  zip(paths, _get_feature(paths, self.feature_names, self.preprocess))})
            semg_feature = self.memo[path]
            semg_feature = semg_feature.copy()

        if imu_root is not None:
            path = self.get_imu_path(imu_root, combo)
            if path not in self.memo:
                logger.debug('Load subject imu data {}', combo.subject)
                paths = [self.get_imu_path(imu_root, Combo(combo.subject, gesture, trial))
                         for gesture, trial in self.gesture_and_trials]
                self.memo.update({path: imu_data for path, imu_data in
                                  zip(paths, _get_imu(paths, self.imu_preprocess, self.preprocess))})
            imu_data = self.memo[path]
            imu_data = imu_data.copy()

        if semg_root is None:
            #gesture = np.repeat(combo.gesture, len(semg_feature))
            #subject = np.repeat(combo.subject, len(semg_feature))

            if imu_root is None:
                data = [semg_feature]
            else:
                if len(semg_feature) == len(imu_data):
                    pass
                elif len(semg_feature) < len(imu_data):
                    print(len(semg_feature), len(imu_data))
                    data_length = len(semg_feature)
                    logger.info('use first %d samples of semg features for data length balance' % data_length)
                    imu_data = imu_data[:data_length,:,:]
                    assert (len(semg_feature) == len(imu_data))
                elif len(semg_feature) > len(imu_data):
                    print(len(semg_feature), len(imu_data))
                    data_length = len(imu_data)
                    print('use first %d samples of imu data for data length balance' % data_length)
                    semg_feature = semg_feature[:data_length,:,:]
                    assert (len(semg_feature) == len(imu_data))
                data = [semg_feature, imu_data]
            gesture = np.repeat(combo.gesture, len(semg_feature))
            subject = np.repeat(combo.subject, len(semg_feature))
            return Trial(data=data, gesture=gesture, subject=subject)
        else:
            assert (len(semg_data) == len(semg_feature))
            gesture = np.repeat(combo.gesture, len(semg_data))
            subject = np.repeat(combo.subject, len(semg_data))

            if imu_root is None:
                data = [semg_data, semg_feature]
            else:
                logger.info('semg length: {}, imu length: {}'.format(
                    len(semg_data), len(imu_data)))
                assert (len(imu_data) == len(semg_data))
                data = [semg_data, semg_feature, imu_data]
            return Trial(data=data, gesture=gesture, subject=subject)


@utils.cached
def _get_data(paths, preprocess):
    return [_get_data_aux(path, preprocess) for path in paths]


def _get_data_aux(path, preprocess):
    data = sio.loadmat(path)['data'].astype(np.float32)
    if preprocess:
        data = preprocess(data, **PREPROCESS_KARGS)

    chnum = data.shape[1]
    data = get_segments(data, window=WINDOW, stride=STRIDE)
    data = data.reshape(-1, WINDOW, chnum)

    return data


def _get_imu(paths, imu_preprocess, preprocess):
    return [_get_imu_aux(path, imu_preprocess, preprocess) for path in paths]

def _get_imu_aux(path, imu_preprocess, preprocess):
    logger.debug('Load {}', path)
    path = path + '.mat'
    mat = sio.loadmat(path)

    if 'data' in mat.keys():
        data = sio.loadmat(path)['data'].astype(np.float32)
    else:
        data = []
        if 'acc' in mat.keys():
            data.append(mat['acc'])
        if 'gyro' in mat.keys():
            data.append(mat['gyro'])
        if 'mag' in mat.keys():
            data.append(mat['mag'])
        data = np.concatenate(data, axis=-1)

    if imu_preprocess:
        # Mainly for downsample
        data = imu_preprocess(data)

    chnum = data.shape[1]
    data = get_segments(data, window=WINDOW, stride=STRIDE)
    data = data.reshape(-1, WINDOW, chnum)

    if preprocess:
        data = preprocess(data)
    return data

def _get_feature(paths, feature_names, preprocess):
    return [_get_feature_aux(path, feature_names, preprocess) for path in paths]


def _get_feature_aux(path, feature_names, preprocess):
    logger.debug('Load {}', path)

    data = []
    for i in range(len(feature_names)):
        input_dir = path + '_' + feature_names[i] + '.mat'
        mat = sio.loadmat(input_dir)
        tmp = mat['data'].astype(np.float32)
        where_are_nan = np.isnan(tmp)
        where_are_inf = np.isinf(tmp)
        tmp[where_are_nan] = 0
        tmp[where_are_inf] = 0
        data.append(tmp)

    data = np.concatenate(data, axis=1)
    if preprocess:
        # Mainly for downsample
        data = preprocess(data)
    #data = data[::10].copy()
    return data


def _get_imufeature(paths, imu_type):
    return [_get_imufeature_aux(path, imu_type) for path in paths]


def _get_imufeature_aux(path, imu_type):
    logger.debug('Load {}', path)
    input_dir = path + '.mat'
    mat = sio.loadmat(input_dir)
    data = mat['data'].astype(np.float32)
    where_are_nan = np.isnan(data)
    where_are_inf = np.isinf(data)
    data[where_are_nan] = 0
    data[where_are_inf] = 0

    return data


def get_segments(data, window, stride):
    return windowed_view(
        data.flat,
        window * data.shape[1],
        (window - stride) * data.shape[1])


def windowed_view(arr, window, overlap):
    from numpy.lib.stride_tricks import as_strided
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step, window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) + arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)



