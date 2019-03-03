import os
import sys
import numpy as np
import sklearn.preprocessing as sp


def mmx_scale(raw_samples, min, max):
    mmx_samples = raw_samples.copy()
    cols = mmx_samples.shape[1]
    for col in range(cols):
        col_samples = mmx_samples[:, col]
        col_min = col_samples.min()
        col_max = col_samples.max()
        k, b = np.linalg.lstsq(
            np.array([[col_min, 1], [col_max, 1]]),
            np.array([min, max]))[0]
        col_samples *= k
        col_samples += b
    return mmx_samples


def main(argc, argv, envp):
    raw_samples = np.array([
        [3, -1.5,  2,   -5.4],
        [0,  4,   -0.3,  2.1],
        [1,  3.3, -1.9, -4.3]])
    print(raw_samples)
    mmx_samples = mmx_scale(raw_samples, 0, 1)
    print(mmx_samples)
    mmx = sp.MinMaxScaler(feature_range=(0, 1))
    mmx_samples = mmx.fit_transform(raw_samples)
    print(mmx_samples)
    return 0

if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
