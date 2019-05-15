from __future__ import division
import re
import numpy as np


def div(up, down):
    try:
        return up / down
    except:
        return np.nan


def parse_log(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    res = [re.compile('.*Epoch\[(\d+)\] Train-accuracy(?:\[g\])?=([.\d]+)'),
           re.compile('.*Epoch\[(\d+)\] Validation-accuracy(?:\[g\])?=([.\d]+)'),
           re.compile('.*Epoch\[(\d+)\] Time.*=([.\d]+)')]

    data = {}
    for l in lines:
        i = 0
        for r in res:
            m = r.match(l)
            if m is not None:
                break
            i += 1
        if m is None:
            continue

        assert len(m.groups()) == 2
        epoch = int(m.groups()[0])
        val = float(m.groups()[1])

        if epoch not in data:
            data[epoch] = [0] * len(res) * 2

        data[epoch][i*2] += val
        data[epoch][i*2+1] += 1

    df = []
    for k, v in data.items():
        try:
            df.append({
                'epoch': k + 1,
                'train': div(v[0], v[1]),
                'val': div(v[2], v[3]),
                'time': div(v[4], v[5])
            })
        except:
            pass

    import pandas as pd
    return pd.DataFrame(df)
