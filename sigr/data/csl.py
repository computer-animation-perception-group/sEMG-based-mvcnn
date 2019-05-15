from __future__ import division
from . import get_data, Combo, Trial
from .. import ROOT, Context, constant
import os
from itertools import product
from functools import partial
import scipy.io as sio
import numpy as np
from logbook import Logger
import joblib as jb
from ..utils import cached
from nose.tools import assert_is_not_none


ROOT = os.path.join(ROOT, '.cache/csl')
NUM_TRIAL = 10
SUBJECTS = list(range(1, 6))
SESSIONS = list(range(1, 6))
NUM_SESSION = len(SESSIONS)
NUM_SUBJECT = len(SUBJECTS)
NUM_SUBJECT_AND_SESSION = len(SUBJECTS) * NUM_SESSION
SUBJECT_AND_SESSIONS = list(range(1, NUM_SUBJECT_AND_SESSION + 1))
GESTURES = list(range(27))
REST_TRIALS = [x - 1 for x in [2, 4, 7, 8, 11, 13, 19, 25, 26, 30]]
NUM_SEMG_ROW = 24
NUM_SEMG_COL = 7
FRAMERATE = 2048
framerate = FRAMERATE
TRIALS = list(range(NUM_TRIAL))
PREPROCESS_KARGS = dict(
    framerate=FRAMERATE,
    num_semg_row=NUM_SEMG_ROW,
    num_semg_col=NUM_SEMG_COL
)

logger = Logger('csl')


def get_general_data(batch_size, adabn, minibatch, downsample, **kargs):
    get_trial = GetTrial(downsample=downsample)
    load = partial(get_data,
                   framerate=FRAMERATE,
                   root=ROOT,
                   last_batch_handle='pad',
                   get_trial=get_trial,
                   batch_size=batch_size,
                   num_semg_row=NUM_SEMG_ROW,
                   num_semg_col=NUM_SEMG_COL)
    train = load(combos=get_combos(product(SUBJECT_AND_SESSIONS, GESTURES[1:], range(0, NUM_TRIAL, 2)),
                                   product(SUBJECT_AND_SESSIONS, GESTURES[:1], REST_TRIALS[0::2])),
                 adabn=adabn,
                 shuffle=True,
                 random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                 random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                 random_shift_vertical=kargs.get('random_shift_vertical', 0),
                 mini_batch_size=batch_size // (NUM_SUBJECT_AND_SESSION if minibatch else 1))
    logger.debug('Training set loaded')
    val = load(combos=get_combos(product(SUBJECT_AND_SESSIONS, GESTURES[1:], range(1, NUM_TRIAL, 2)),
                                 product(SUBJECT_AND_SESSIONS, GESTURES[:1], REST_TRIALS[1::2])),
               shuffle=False)
    logger.debug('Test set loaded')
    return train, val


def get_intra_session_val(fold, batch_size, preprocess, **kargs):
    get_trial = GetTrial(preprocess=preprocess)
    load = partial(get_data,
                   amplitude_weighting=kargs.get('amplitude_weighting', False),
                   amplitude_weighting_sort=kargs.get('amplitude_weighting_sort', False),
                   framerate=FRAMERATE,
                   root=ROOT,
                   last_batch_handle='pad',
                   get_trial=get_trial,
                   batch_size=batch_size,
                   num_semg_row=NUM_SEMG_ROW,
                   num_semg_col=NUM_SEMG_COL,
                   random_state=np.random.RandomState(42))
    subject = fold // (NUM_SESSION * NUM_TRIAL) + 1
    session = fold // NUM_TRIAL % NUM_SESSION + 1
    fold = fold % NUM_TRIAL
    val = load(combos=get_combos(product([encode_subject_and_session(subject, session)],
                                         GESTURES[1:], [fold]),
                                 product([encode_subject_and_session(subject, session)],
                                         GESTURES[:1], REST_TRIALS[fold:fold + 1])),
               shuffle=False)
    return val


def get_universal_intra_session_data(fold, batch_size, preprocess, balance_gesture, **kargs):
    get_trial = GetTrial(preprocess=preprocess)
    load = partial(get_data,
                   amplitude_weighting=kargs.get('amplitude_weighting', False),
                   amplitude_weighting_sort=kargs.get('amplitude_weighting_sort', False),
                   framerate=FRAMERATE,
                   root=ROOT,
                   last_batch_handle='pad',
                   get_trial=get_trial,
                   batch_size=batch_size,
                   num_semg_row=NUM_SEMG_ROW,
                   num_semg_col=NUM_SEMG_COL)
    trial = fold
    train = load(combos=get_combos(product(SUBJECT_AND_SESSIONS,
                                           GESTURES[1:], [i for i in range(NUM_TRIAL) if i != trial]),
                                   product(SUBJECT_AND_SESSIONS,
                                           GESTURES[:1], [REST_TRIALS[i] for i in range(NUM_TRIAL) if i != trial])),
                 balance_gesture=balance_gesture,
                 random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                 random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                 random_shift_vertical=kargs.get('random_shift_vertical', 0),
                 shuffle=True)
    assert_is_not_none(train)
    logger.debug('Training set loaded')
    val = load(combos=get_combos(product(SUBJECT_AND_SESSIONS,
                                         GESTURES[1:], [trial]),
                                 product(SUBJECT_AND_SESSIONS,
                                         GESTURES[:1], REST_TRIALS[trial:trial + 1])),
               shuffle=False)
    logger.debug('Test set loaded')
    assert_is_not_none(val)
    return train, val


def get_intra_session_data(fold, batch_size, preprocess, balance_gesture, **kargs):
    get_trial = GetTrial(preprocess=preprocess)
    load = partial(get_data,
                   amplitude_weighting=kargs.get('amplitude_weighting', False),
                   amplitude_weighting_sort=kargs.get('amplitude_weighting_sort', False),
                   framerate=FRAMERATE,
                   root=ROOT,
                   last_batch_handle='pad',
                   get_trial=get_trial,
                   batch_size=batch_size,
                   num_semg_row=NUM_SEMG_ROW,
                   num_semg_col=NUM_SEMG_COL)
    subject = fold // (NUM_SESSION * NUM_TRIAL) + 1
    session = fold // NUM_TRIAL % NUM_SESSION + 1
    fold = fold % NUM_TRIAL
    train = load(combos=get_combos(product([encode_subject_and_session(subject, session)],
                                           GESTURES[1:], [f for f in range(NUM_TRIAL) if f != fold]),
                                   product([encode_subject_and_session(subject, session)],
                                           GESTURES[:1], [REST_TRIALS[f] for f in range(NUM_TRIAL) if f != fold])),
                 balance_gesture=balance_gesture,
                 random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                 random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                 random_shift_vertical=kargs.get('random_shift_vertical', 0),
                 shuffle=True)
    assert_is_not_none(train)
    logger.debug('Training set loaded')
    val = load(combos=get_combos(product([encode_subject_and_session(subject, session)],
                                         GESTURES[1:], [fold]),
                                 product([encode_subject_and_session(subject, session)],
                                         GESTURES[:1], REST_TRIALS[fold:fold + 1])),
               shuffle=False)
    logger.debug('Test set loaded')
    assert_is_not_none(val)
    return train, val


def get_inter_session_data(fold, batch_size, preprocess, adabn, minibatch, balance_gesture, **kargs):
    #  TODO: calib
    get_trial = GetTrial(preprocess=preprocess)
    load = partial(get_data,
                   framerate=FRAMERATE,
                   root=ROOT,
                   last_batch_handle='pad',
                   get_trial=get_trial,
                   batch_size=batch_size,
                   num_semg_row=NUM_SEMG_ROW,
                   num_semg_col=NUM_SEMG_COL)
    subject = fold // NUM_SESSION + 1
    session = fold % NUM_SESSION + 1
    train = load(combos=get_combos(product([encode_subject_and_session(subject, i) for i in SESSIONS if i != session],
                                           GESTURES[1:], TRIALS),
                                   product([encode_subject_and_session(subject, i) for i in SESSIONS if i != session],
                                           GESTURES[:1], REST_TRIALS)),
                 adabn=adabn,
                 mini_batch_size=batch_size // (NUM_SESSION - 1 if minibatch else 1),
                 balance_gesture=balance_gesture,
                 random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                 random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                 random_shift_vertical=kargs.get('random_shift_vertical', 0),
                 shuffle=True)
    logger.debug('Training set loaded')
    val = load(combos=get_combos(product([encode_subject_and_session(subject, session)],
                                         GESTURES[1:], TRIALS),
                                 product([encode_subject_and_session(subject, session)],
                                         GESTURES[:1], REST_TRIALS)),
               shuffle=False)
    logger.debug('Test set loaded')
    return train, val


def get_inter_session_val(fold, batch_size, preprocess, **kargs):
    #  TODO: calib
    get_trial = GetTrial(preprocess=preprocess)
    load = partial(get_data,
                   framerate=FRAMERATE,
                   root=ROOT,
                   last_batch_handle='pad',
                   get_trial=get_trial,
                   batch_size=batch_size,
                   num_semg_row=NUM_SEMG_ROW,
                   num_semg_col=NUM_SEMG_COL,
                   random_state=np.random.RandomState(42))
    subject = fold // NUM_SESSION + 1
    session = fold % NUM_SESSION + 1
    val = load(combos=get_combos(product([encode_subject_and_session(subject, session)],
                                         GESTURES[1:], TRIALS),
                                 product([encode_subject_and_session(subject, session)],
                                         GESTURES[:1], REST_TRIALS)),
               shuffle=False)
    return val


def get_universal_inter_session_data(fold, batch_size, preprocess, adabn, minibatch, balance_gesture, **kargs):
    #  TODO: calib
    get_trial = GetTrial(preprocess=preprocess)
    load = partial(get_data,
                   framerate=FRAMERATE,
                   root=ROOT,
                   last_batch_handle='pad',
                   get_trial=get_trial,
                   batch_size=batch_size,
                   num_semg_row=NUM_SEMG_ROW,
                   num_semg_col=NUM_SEMG_COL)
    session = fold + 1
    train = load(combos=get_combos(product([encode_subject_and_session(s, i) for s, i in
                                            product(SUBJECTS, [i for i in SESSIONS if i != session])],
                                           GESTURES[1:], TRIALS),
                                   product([encode_subject_and_session(s, i) for s, i in
                                            product(SUBJECTS, [i for i in SESSIONS if i != session])],
                                           GESTURES[:1], REST_TRIALS)),
                 adabn=adabn,
                 mini_batch_size=batch_size // (NUM_SUBJECT * (NUM_SESSION - 1) if minibatch else 1),
                 balance_gesture=balance_gesture,
                 random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                 random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                 random_shift_vertical=kargs.get('random_shift_vertical', 0),
                 shuffle=True)
    logger.debug('Training set loaded')
    val = load(combos=get_combos(product([encode_subject_and_session(s, session) for s in SUBJECTS],
                                         GESTURES[1:], TRIALS),
                                 product([encode_subject_and_session(s, session) for s in SUBJECTS],
                                         GESTURES[:1], REST_TRIALS)),
               adabn=adabn,
               mini_batch_size=batch_size // (NUM_SUBJECT if minibatch else 1),
               shuffle=False)
    logger.debug('Test set loaded')
    return train, val


def get_intra_subject_data(fold, batch_size, cut, bandstop, adabn, minibatch, **kargs):
    get_trial = GetTrial(cut=cut, bandstop=bandstop)
    load = partial(get_data,
                   framerate=FRAMERATE,
                   root=ROOT,
                   last_batch_handle='pad',
                   get_trial=get_trial,
                   batch_size=batch_size,
                   num_semg_row=NUM_SEMG_ROW,
                   num_semg_col=NUM_SEMG_COL)
    subject = fold // NUM_TRIAL + 1
    fold = fold % NUM_TRIAL
    train = load(combos=get_combos(product([encode_subject_and_session(subject, session) for session in SESSIONS],
                                           GESTURES[1:], [f for f in range(NUM_TRIAL) if f != fold]),
                                   product([encode_subject_and_session(subject, session) for session in SESSIONS],
                                           GESTURES[:1], [REST_TRIALS[f] for f in range(NUM_TRIAL) if f != fold])),
                 adabn=adabn,
                 mini_batch_size=batch_size // (NUM_SESSION if minibatch else 1),
                 random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                 random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                 random_shift_vertical=kargs.get('random_shift_vertical', 0),
                 shuffle=True)
    logger.debug('Training set loaded')
    val = load(combos=get_combos(product([encode_subject_and_session(subject, session) for session in SESSIONS],
                                         GESTURES[1:], [fold]),
                                 product([encode_subject_and_session(subject, session) for session in SESSIONS],
                                         GESTURES[:1], REST_TRIALS[fold:fold + 1])),
               shuffle=False)
    logger.debug('Test set loaded')
    return train, val


class GetTrial(object):

    def __init__(self, preprocess=None):
        self.preprocess = preprocess
        self.memo = {}

    def __call__(self, root, combo):
        subject, session = decode_subject_and_session(combo.subject)
        path = os.path.join(root,
                            'subject%d' % subject,
                            'session%d' % session,
                            'gest%d.mat' % combo.gesture)
        if path not in self.memo:
            data = _get_data(path, self.preprocess)
            self.memo[path] = data
            logger.debug('{}', path)
        else:
            data = self.memo[path]
        assert combo.trial < len(data), str(combo)
        data = data[combo.trial].copy()
        gesture = np.repeat(combo.gesture, len(data))
        subject = np.repeat(combo.subject, len(data))
        return Trial(data=data, gesture=gesture, subject=subject)


@cached
def _get_data(path, preprocess):
    data = sio.loadmat(path)['gestures']
    data = [np.transpose(np.delete(segment.astype(np.float32), np.s_[7:192:8], 0))
            for segment in data.flat]
    if preprocess:
        data = list(Context.parallel(jb.delayed(preprocess)(segment, **PREPROCESS_KARGS)
                                     for segment in data))
    return data


#  @cached
#  def _get_data(path, bandstop, cut, downsample):
    #  data = sio.loadmat(path)['gestures']
    #  data = [np.transpose(np.delete(segment.astype(np.float32), np.s_[7:192:8], 0))
            #  for segment in data.flat]
    #  if bandstop:
        #  data = list(Context.parallel(jb.delayed(get_bandstop)(segment) for segment in data))
    #  if cut is not None:
        #  data = list(Context.parallel(jb.delayed(cut)(segment, framerate=FRAMERATE) for segment in data))
    #  if downsample > 1:
        #  data = [segment[::downsample].copy() for segment in data]
    #  return data


def decode_subject_and_session(ss):
    return (ss - 1) // NUM_SESSION + 1, (ss - 1) % NUM_SESSION + 1


def encode_subject_and_session(subject, session):
    return (subject - 1) * NUM_SESSION + session


def get_bandstop(data):
    from ..utils import butter_bandstop_filter
    return np.array([butter_bandstop_filter(ch, 45, 55, 2048, 2) for ch in data])


def get_combos(*args):
    for arg in args:
        if isinstance(arg, tuple):
            arg = [arg]
        for a in arg:
            combo = Combo(*a)
            if ignore_missing(combo):
                continue
            yield combo


def ignore_missing(combo):
    return combo.subject == 19 and combo.gesture in (8, 9) and combo.trial == 9
