"""
This is a ninapro-db6 dataset
"""
from .ninapro_dataset import Dataset as Base
from ... import constant
from logbook import Logger


logger = Logger(__name__)

WINDOW = 20
STRIDE = 1

class Dataset(Base):

    name = 'ninapro-db6'

    num_semg_row = 1
    num_semg_col = 14
    num_imu_row = 1
    num_imu_col = 14

    feature_extraction_winlength = WINDOW
    feature_extraction_winstride = STRIDE

    feature_names = ['dwpt', 'dwt', 'mav','wl','wamp','mavslpframewise','arc','mnf_MEDIAN_POWER','psr']

    #semg_root = '/code/dqf/data/ninapro-db6/semg'
    semg_root=None
    feature_root = '/feature'
    imu_root = '/imu'

    subjects = list(range(10))
    subjects.remove(8) # Subject 8 , semg data lack a channel, so we decide to ignore it
    subjects.remove(1) # Subject 1, just consists of 108 trials which is less than others by 12 trials
    gestures = list(range(7))
    trials = list(range(120))

    def get_one_fold_intra_subject_trials(self):
        return self.trials[::2], self.trials[1::2]



