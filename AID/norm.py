import os
import sys
import numpy as np
import sklearn.preprocessing as sp


def normalize(raw_samples):
    nor_samples = raw_samples.copy()
    rows = nor_samples.shape[0]
    for row in range(rows):
        row_samples = nor_samples[row]
        row_abs = abs(row_samples)
        row_abs_sum = row_abs.sum()
        row_samples /= row_abs_sum
    return nor_samples


def main(argc, argv, envp):
    raw_samples = np.array([
        [3, -1.5,  2,   -5.4],
        [0,  4,   -0.3,  2.1],
        [1,  3.3, -1.9, -4.3]])
    print(raw_samples)
    nor_samples = normalize(raw_samples)
    print(nor_samples)
    for row in range(nor_samples.shape[0]):
        row_samples = nor_samples[row]
        abs_samples = abs(row_samples)
        sum_samples = abs_samples.sum()
        print(sum_samples)
    nor_samples = sp.normalize(raw_samples, norm='l1')
    print(nor_samples)
    for row in range(nor_samples.shape[0]):
        row_samples = nor_samples[row]
        abs_samples = abs(row_samples)
        sum_samples = abs_samples.sum()
        print(sum_samples)
    return 0

if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
