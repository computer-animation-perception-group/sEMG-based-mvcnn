from .ninapro_dataset import Dataset as Base
from ... import constant
from logbook import Logger


logger = Logger(__name__)

WINDOW = constant.FEATURE_EXTRACTION_WIN_LEN
STRIDE = constant.FEATURE_EXTRACTION_WIN_STRIDE
USE_IMU = constant.USE_IMU
# Need to switch to False for single-view
IFMULTI = constant.IFMULTIVIEW


class Dataset(Base):
    name = 'ninapro-db1'

    num_semg_row = 1
    num_semg_col = 10

    feature_extraction_winlength = WINDOW
    feature_extraction_winstride = STRIDE

    feature_names = constant.FEATURE_LIST

    #semg_root = '/home/weiwentao/public-2/data/ninapro-db1/data' if IFMULTI else None
    semg_root = None

    feature_root = '/feature'

    # NinaPro DB1 doesn't have IMU Data
    imu_root = None

    subjects = list(range(27))
    gestures = list(range(1, 53))
    trials = list(range(10))




