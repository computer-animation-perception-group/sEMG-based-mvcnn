from __future__ import division
import numpy as np
import joblib as jb
from nose.tools import assert_greater
from .utils import return_list, cached
from . import Context


def get_vote_accuracy_curve(labels, predictions, segments, windows, balance=False):
    if len(set(segments)) < len(windows):
        func = get_vote_accuracy_curve_aux
    else:
        func = get_vote_accuracy_curve_aux_few_windows
    return func(np.asarray(labels),
                np.asarray(predictions),
                np.asarray(segments),
                np.asarray(windows),
                balance)


@cached
def get_vote_accuracy_curve_aux(labels, predictions, segments, windows, balance):
    segment_labels = partial_vote(labels, segments)
    return (
        np.asarray(windows),
        np.array(list(Context.parallel(
            jb.delayed(get_vote_accuracy_curve_step)(
                segment_labels,
                predictions,
                segments,
                window,
                balance
            ) for window in windows
        )))
    )


@cached
def get_vote_accuracy_curve_aux_few_windows(labels, predictions, segments, windows, balance):
    segment_labels = partial_vote(labels, segments)
    return (
        np.asarray(windows),
        np.array([
            get_vote_accuracy_curve_step(
                segment_labels,
                predictions,
                segments,
                window,
                balance,
                parallel=True
            ) for window in windows
        ])
    )


def get_vote_accuracy(labels, predictions, segments, window, balance):
    _, y = get_vote_accuracy_curve(labels, predictions, segments, [window], balance)
    return y[0]


vote = get_vote_accuracy


def get_segment_vote_accuracy(segment_label, segment_predictions, window):
    def gen():
        count = {
            label: np.hstack([[0], np.cumsum(segment_predictions == label)])
            for label in set(segment_predictions)
        }
        tmp = window
        if tmp == -1:
            tmp = len(segment_predictions)
        tmp = min(tmp, len(segment_predictions))
        for begin in range(len(segment_predictions) - tmp + 1):
            yield segment_label == max(
                count,
                key=lambda label: count[label][begin + tmp] - count[label][begin]
            ), segment_label
    return list(gen())


def get_vote_accuracy_curve_step(segment_labels, predictions, segments, window,
                                 balance,
                                 parallel=False):
    def gen():
        #  assert_greater(window, 0)
        assert window > 0 or window == -1
        if not parallel:
            for segment_label, segment_predictions in zip(segment_labels, split(predictions, segments)):
                for ret in get_segment_vote_accuracy(segment_label, segment_predictions, window):
                    yield ret
        else:
            for rets in Context.parallel(
                jb.delayed(get_segment_vote_accuracy)(segment_label, segment_predictions, window)
                for segment_label, segment_predictions in zip(segment_labels, split(predictions, segments))
            ):
                for ret in rets:
                    yield ret

    good, labels = zip(*list(gen()))
    good = np.asarray(good)

    if not balance:
        return np.sum(good) / len(good)
    else:
        acc = []
        for label in set(labels):
            mask = [labels == label]
            acc.append(np.sum(good[mask]) / np.sum(mask))
        return np.mean(acc)


@return_list
def partial_vote(labels, segments, length=None):
    for part in split(labels, segments):
        part = list(part)

        if length is not None:
            part = part[:length]

        assert_greater(len(part), 0)
        yield max([(part.count(label), label) for label in set(part)])[1]


def split(labels, segments):
    return [labels[segments == segment] for segment in sorted(set(segments))]
