from .ninapro_dataset import Dataset as Base
from ... import constant
from logbook import Logger
import math
from functools import partial
from itertools import product


logger = Logger(__name__)

WINDOW = constant.FEATURE_EXTRACTION_WIN_LEN
STRIDE = constant.FEATURE_EXTRACTION_WIN_STRIDE


class Dataset(Base):

    name = 'ninapro-db2'

    num_semg_row = 1
    num_semg_col = 12
    num_imu_row = 1
    num_imu_col = 12

    feature_extraction_winlength = WINDOW
    feature_extraction_winstride = STRIDE

    feature_names = constant.FEATURE_LIST

    #semg_root = '/home/weiwentao/public-2/data/ninapro-db2-var-raw-noflip/data'
    semg_root = None
    feature_root = '/feature'
    #if constant.USE_IMU:
    imu_root = '/imu'
    #else:
    #    imu_root = None

    subjects = list(range(4))  # truth is 40
    gestures = list(range(50)) # truth is 50
    trials = list(range(6))


    def get_5_fold_inter_subject_data(self, fold, batch_size, preprocess, imu_preprocess,
                                      adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, imu_type, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names,
                                        imu_type=imu_type)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        folds = self.get_k_fold_inter_subject(5)
        val_subjects = folds[fold]
        train_subjects = [i for i in self.subject if i not in val_subjects]
        train = load(
            combos=self.get_combos(product(train_subjects, self.gestures, self.trials)),
            adabn=adabn,
            mini_batch_size=batch_size // (5 if minibatch else 1),
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

    def get_10_fold_inter_subject_data(self, fold, batch_size, preprocess, imu_preprocess,
                                      adabn, minibatch, feature_name, window, num_semg_row, num_semg_col, imu_type, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials,
                                        preprocess=preprocess,
                                        imu_preprocess=imu_preprocess,
                                        feature_names=self.feature_names,
                                        imu_type=imu_type)
        load = partial(self.get_dataiter,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size)
        folds = self.get_k_fold_inter_subject(10)
        val_subjects = folds[fold]
        train_subjects = [i for i in self.subject if i not in val_subjects]
        train = load(
            combos=self.get_combos(product(train_subjects, self.gestures, self.trials)),
            adabn=adabn,
            mini_batch_size=batch_size // (10 if minibatch else 1),
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

    def get_one_fold_intra_subject_trials(self):
        return [0, 2, 3, 5], [1, 4]
        #return [0], [1] # for test



