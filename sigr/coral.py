import numpy as np
import scipy.linalg as splg


def get_coral_params(ds, dt, lam=1e-3):
    ms = ds.mean(axis=0)
    ds = ds - ms
    mt = dt.mean(axis=0)
    dt = dt - mt
    cs = np.cov(ds.T) + lam * np.eye(ds.shape[1])
    ct = np.cov(dt.T) + lam * np.eye(dt.shape[1])
    sqrt = splg.sqrtm
    w = sqrt(ct).dot(np.linalg.inv(sqrt(cs)))
    b = mt - w.dot(ms.reshape(-1, 1)).ravel()
    return w, b
