import os
import sys
import numpy as np
import sklearn.preprocessing as sp


def binarize(raw_samples, threshold):
    bin_samples = raw_samples.copy()
    bin_samples[bin_samples <= threshold] = 0
    bin_samples[bin_samples > threshold] = 1
    return bin_samples


def main(argc, argv, envp):
    raw_samples = np.array([
        [3, -1.5,  2,   -5.4],
        [0,  4,   -0.3,  2.1],
        [1,  3.3, -1.9, -4.3]])
    print(raw_samples)
    bin_samples = binarize(raw_samples, 1.4)
    print(bin_samples)
    bin = sp.Binarizer(threshold=1.4)
    bin_samples = bin.transform(raw_samples)
    print(bin_samples)
    return 0

if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
