import os
import sys
import numpy as np
import sklearn.preprocessing as sp


def std_scale(raw_samples):
    std_samples = raw_samples.copy()
    cols = std_samples.shape[1]
    for col in range(cols):
        col_samples = std_samples[:, col]
        col_mean = col_samples.mean()
        col_std = col_samples.std()
        col_samples -= col_mean
        col_samples /= col_std
    return std_samples


def main(argc, argv, envp):
    raw_samples = np.array([
        [3, -1.5,  2,   -5.4],
        [0,  4,   -0.3,  2.1],
        [1,  3.3, -1.9, -4.3]])
    print(raw_samples)
    raw_means = raw_samples.mean(axis=0)
    print(raw_means)
    std_samples = std_scale(raw_samples)
    std_means = std_samples.mean(axis=0)
    print(std_means)
    std_stds = std_samples.std(axis=0)
    print(std_stds)
    std_samples = sp.scale(raw_samples)
    std_means = std_samples.mean(axis=0)
    print(std_means)
    std_stds = std_samples.std(axis=0)
    print(std_stds)
    return 0

if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
