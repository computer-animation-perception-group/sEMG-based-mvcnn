from itertools import product, starmap
from . import get_data, Combo
from .. import ROOT
import os
import numpy as np


ROOT = os.path.join(ROOT, '.cache/mat.s21.bandstop-45-55.s1000m.scale-01')


def get_coral(folds, batch_size):
    subjects = [1, 3, 6, 8, 10, 12, 14, 16, 18, 20]
    return get_data(
        root=ROOT,
        # combos=get_combos(product([subjects[fold] for fold in folds], [100, 101], [0])),
        combos=get_combos(product([subjects[fold] for fold in folds], range(1, 9), [0])),
        mean=0.5,
        scale=2,
        batch_size=2000,
        last_batch_handle='pad',
        shuffle=False,
        adabn=True
    )


def get_combos(prods):
    return list(starmap(Combo, prods))


def get_stats():
    subjects = [1, 3, 6, 8, 10, 12, 14, 16, 18, 20]
    load = lambda subject: get_data(
        root=ROOT,
        combos=get_combos(product([subject], range(1, 9), range(10))),
        mean=0.5,
        scale=2,
        batch_size=1000,
        last_batch_handle='roll_over'
    )
    stats = []
    for subject in subjects:
        batch = next(load(subject)[0])
        data = batch.data[0].asnumpy()
        stats.append({
            'std': data.std()
        })
    import pandas as pd
    return pd.DataFrame(stats, index=range(10))


def get_general_data(root, batch_size, with_subject):
    subjects = [1, 3, 6, 8, 10, 12, 14, 16, 18, 20]
    load = lambda **kargs: get_data(
        root=root,
        mean=0.5,
        scale=2,
        with_subject=with_subject,
        batch_size=batch_size,
        last_batch_handle='roll_over',
        **kargs
    )
    val, num_val = load(combos=get_combos(product(subjects, range(1, 9), range(1, 10, 2))))
    train, num_train = load(combos=get_combos(product(subjects, range(1, 9), range(0, 10, 2))))
    return train, val, num_train, num_val


def get_inter_subject_data(
    root,
    fold,
    batch_size,
    maxforce,
    target_binary,
    calib,
    with_subject,
    with_target_gesture,
    random_scale,
    random_bad_channel,
    shuffle,
    adabn,
    window,
    only_calib,
    soft_label,
    minibatch,
    fft,
    fft_append,
    dual_stream,
    lstm,
    dense_window,
    lstm_window
):
    subjects = [1, 3, 6, 8, 10, 12, 14, 16, 18, 20]

    num_subject = 10 if maxforce or calib else 9
    if minibatch:
        assert batch_size % num_subject == 0, '%d %% %d' % (batch_size, num_subject)
        mini_batch_size = batch_size // num_subject
    else:
        mini_batch_size = batch_size

    load = lambda **kargs: get_data(
        root=root,
        mean=0.5,
        scale=2,
        with_subject=with_subject,
        target_binary=target_binary,
        batch_size=batch_size,
        with_target_gesture=with_target_gesture,
        fft=fft,
        fft_append=fft_append,
        dual_stream=dual_stream,
        **kargs
    )
    val_subject = subjects[fold]
    del subjects[fold]
    val = load(
        combos=get_combos(product([val_subject], range(1, 9), range(1, 10) if calib else range(10))),
        last_batch_handle='pad',
        shuffle=False,
        window=(window // (lstm_window or window)) if lstm else window,
        num_ignore_per_segment=window - 1 if lstm else 0,
        dense_window=dense_window
    )

    if maxforce and calib:
        target_combos = get_combos(product([val_subject], list(range(1, 9)) * 10 + [100, 101], [0] * (9 if target_binary else 1)))
    elif maxforce:
        target_combos = get_combos(product([val_subject], [100, 101], [0] * 41 * (9 if target_binary else 1)))
    elif only_calib:
        target_combos = get_combos(product([val_subject], list(range(1, 9)), [0]))
    elif calib:
        target_combos = get_combos(product([val_subject], list(range(1, 9)) * 10, [0] * (9 if target_binary else 1)))
    else:
        target_combos = None

    if only_calib:
        combos = []
    else:
        combos = get_combos(product(subjects, range(1, 9), range(10)))
        if maxforce:
            combos += get_combos(product(subjects, [100, 101], [0]))

    if soft_label:
        import pandas as pd
        soft_label = pd.DataFrame.from_csv(os.path.join(os.path.dirname(__file__), 's21_soft_label.scv'))

    train = load(
        combos=combos,
        target_combos=target_combos,
        random_scale=random_scale,
        random_bad_channel=random_bad_channel,
        last_batch_handle='pad',
        shuffle=shuffle,
        mini_batch_size=mini_batch_size,
        soft_label=False if soft_label is False else soft_label[soft_label['fold'] == fold][[str(i) for i in range(8)]].as_matrix(),
        adabn=adabn,
        window=window,
        dense_window=dense_window
    )
    return train, val


def get_inter_subject_val(fold, batch_size, calib, **kargs):
    subjects = [1, 3, 6, 8, 10, 12, 14, 16, 18, 20]
    return get_data(
        combos=get_combos(product([subjects[fold]], range(1, 9), range(1, 10) if calib else range(10))),
        root=ROOT,
        mean=0.5,
        scale=2,
        batch_size=batch_size,
        last_batch_handle='pad',
        shuffle=False,
        random_state=np.random.RandomState(42),
        **kargs
    )


def get_inter_subject_train(fold, batch_size):
    subjects = [1, 3, 6, 8, 10, 12, 14, 16, 18, 20]
    return get_data(
        combos=get_combos(product([subjects[i] for i in range(10) if i != fold], range(1, 9), range(10))),
        root=ROOT,
        mean=0.5,
        scale=2,
        batch_size=batch_size,
        last_batch_handle='pad',
        shuffle=False
    )
