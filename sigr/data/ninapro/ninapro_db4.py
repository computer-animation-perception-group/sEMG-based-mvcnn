from .ninapro_dataset import Dataset as Base
from ... import constant
from logbook import Logger


logger = Logger(__name__)

WINDOW = 20
STRIDE = 1


class Dataset(Base):

    name = 'ninapro-db4'

    num_semg_row = 12
    num_semg_col = 1

    feature_extraction_winlength = WINDOW
    feature_extraction_winstride = STRIDE

    feature_names = ['dwpt', 'dwt', 'mav','wl','wamp','mavslpframewise','arc','mnf_MEDIAN_POWER','psr']

    #semg_root = '/code/dqf/data/ninapro-db4/semg/data'
    semg_root = None
    feature_root = '/feature'
    imu_root = None

    subjects = list(range(10))
    gestures = list(range(1, 53))
    trials = list(range(6))

    def get_one_fold_intra_subject_trials(self):
        return [0, 2, 3, 5], [1, 4]
