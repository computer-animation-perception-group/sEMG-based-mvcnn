NUM_LSTM_HIDDEN = 128
NUM_LSTM_LAYER = 1
LSTM_DROPOUT = 0.
FRAMERATE=2000
NUM_SEMG_ROW = 1
NUM_SEMG_COL = 12
NUM_SEMG_POINT = NUM_SEMG_ROW * NUM_SEMG_COL
NUM_IMU_ROW = 1
NUM_IMU_COL = 12
NUM_FILTER = 16
NUM_HIDDEN = 512
NUM_BOTTLENECK = 128
DROPOUT = 0.5
GAMMA = 10
NUM_FEATURE_BLOCK = 2
NUM_GESTURE_BLOCK = 0
NUM_SUBJECT_BLOCK = 0
NUM_PIXEL = 2
LAMBDA_SCALE = 1
NUM_TZENG_BATCH = 2
NUM_ADABN_EPOCH = 1
RANDOM_SHIFT_FILL = 'zero'
NUM_CONV_LAYER = 2
NUM_CONV_FILTER = 64
NUM_LC_LAYER = 2
NUM_LC_HIDDEN = 64
LC_KERNEL = 1
LC_STRIDE = 1
LC_PAD = 0
NUM_FC_LAYER = 2
NUM_FC_HIDDEN = 512
NUM_MINI_BATCH = 1
LR = 0.1
WD = 0.0001
LR_FACTOR = 0.1
#  GLOVE_LOSS_WEIGHT = 1
#  NUM_GLOVE_LAYER = 128
#  NUM_GLOVE_HIDDEN = 128
NUM_EPOCH = 28
LR_STEP = [16, 24]
BATCH_SIZE = 1000
DECAY_ALL = True
SNAPSHOT_PERIOD = 28

FEATURE_EXTRACTION_WIN_LEN = 20
FEATURE_EXTRACTION_WIN_STRIDE = 1
USE_IMU = True
IFMULTIVIEW = False
ACTIVITY_IMAGE_PREPROCESS = 'lowpass'
FEATURE_MAP_PREPROCESS = 'lowpass'
#FEATURE_LIST = ['mav','wl','wamp','mavslpphinyomark','arc','mnf_MEDIAN_POWER','psr']

#FEATURE_LIST = ['iemg','mav','log','rms','vorder','mav1','mav2',
#                'arr_Range', 'arr_H', 'arr_meanraise', 'arr_stdraise', 'arr_mean', 'arr_std', 'arr_Rangen', 'arr_offon', 'arr_offset',
#                'mrwa', 'dwpt_sd', 'dwpt_mean']

#FEATURE_LIST = ['dwpt']

#FEATURE_LIST = ['dwpt_energy','dwpt_kurtosis','dwpt_m2','dwpt_m3','dwpt_m4','dwpt_skewness','dwpt_mean','dwpt_sd']

#FEATURE_LIST = ['fft_power_db1', 'mdf_MEDIAN_POWER', 'mnf_MEDIAN_POWER', 'pkf', 'mnp', 'ttp', 'smn1', 'smn2', 'smn3', 'fr', 'psr', 'vcf', 'hos']

#FEATURE_LIST = ['arr29']

FEATURE_LIST = ['dwpt', 'dwt', 'mav','wl','wamp','mavslpframewise','arc','mnf_MEDIAN_POWER','psr', 'tdd_cor']

#FEATURE_LIST = ['dwpt', 'dwt']


#FEATURE_LIST = ['mav','wl','wamp','mavslpframewise','arc','mnf_MEDIAN_POWER','psr']

#FEATURE_LIST = ['mavslpframewise']
