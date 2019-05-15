import os
import scipy.io as sio
import numpy as np
from scipy import interpolate
subjects = range(10)
gestures = range(1)
trials = range(6)

source_root = '/home/daiqingfeng/10.214.150.99/semg/dqf/data/ninapro-db5/imu/data'

target_root = '/home/daiqingfeng/10.214.150.99/semg/dqf/data/ninapro-db5/expand_imu/data'

for subject in subjects:
    for gesture in gestures:
        for trial in trials:
            source_path = os.path.join(source_root, '{s:03d}', '{g:03d}', '{s:03d}_{g:03d}_{t:03d}.mat').format(s=subject,g=gesture,t=trial)

            imu = sio.loadmat(source_path)['data']

            index = np.array([0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0])

            imu = imu[:,index]

            print('imu shape:', imu.shape)

            target_path = os.path.join(target_root, '{s:03d}', '{g:03d}').format(s=subject,g=gesture)

            filename = os.path.join(target_path, '{s:03d}_{g:03d}_{t:03d}.mat').format(s=subject,g=gesture,t=trial)

            if os.path.isdir(target_path) is False:
                os.makedirs(target_path)

            sio.savemat(filename, {'data':imu})
