from .ninapro_dataset import Dataset as Base
from ... import constant
from logbook import Logger
from nose.tools import assert_equal
from functools import partial
from itertools import product

logger = Logger(__name__)

WINDOW = 300
STRIDE = 100

class Dataset(Base):
    name = 'biopatrec-db4'

    num_semg_row = 1
    num_semg_col = 16

    feature_extraction_winlength = WINDOW
    feature_extraction_winstride = STRIDE

    feature_names = constant.FEATURE_LIST

    semg_root = None
    feature_root = '/feature'
    imu_root =  None

    subjects = list(range(8))
    gestures = list(range(8))
    trials = list(range(3))

    def get_one_fold_intra_subject_trials(self):
        return [0], [1, 2]

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
        train_trials = [0, 1, 2]
        val_trials = [2]
        train = load(
            combos=self.get_combos(product(self.subjects, self.gestures,
                                           [i for i in train_trials])),
            adabn=adabn,
            mini_batch_size=batch_size // (self.num_subject if minibatch else 1),
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


