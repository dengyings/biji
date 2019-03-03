import os
import sys
import numpy as np
import sklearn.preprocessing as sp


def onehot_encode(raw_samples):
    code_tables = []
    for colume in raw_samples.T:
        code_table = {}
        for value in colume:
            code_table[value] = None
        code_tables.append(code_table)

    for code_table in code_tables:
        size = len(code_table)
        for one, key in enumerate(sorted(code_table.keys())):
            code_table[key] = np.zeros(shape=size, dtype=int)
            code_table[key][one] = 1
    ohe_samples = []
    for raw_sample in raw_samples:
        ohe_sample = np.array([], dtype=int)
        for colume, feature in enumerate(raw_sample):
            ohe_sample = np.hstack((
                ohe_sample, code_tables[colume][feature]))
        ohe_samples.append(ohe_sample)
    return np.array(ohe_samples)


def main(argc, argv, envp):
    raw_samples = np.array([
        [0, 0, 3],
        [1, 1, 0],
        [0, 2, 1],
        [1, 0, 2]])
    print(raw_samples)
    ohe_samples = onehot_encode(raw_samples)
    print(ohe_samples)
    ohe = sp.OneHotEncoder(sparse=False, dtype=int)
    ohe_samples = ohe.fit_transform(raw_samples)
    print(ohe_samples)
    return 0

if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
