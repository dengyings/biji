import os
import sys
import numpy as np
import sklearn.decomposition as dc


def make_data():
    a = np.random.normal(size=250)
    b = np.random.normal(size=250)
    c = 2 * a + 3 * b
    d = 4 * a - b
    e = c + 2 * d
    x = np.c_[d, b, e, a, c]
    return x


def train_model(x):
    model = dc.PCA()
    model.fit(x)
    return model


def reduce_model(model, n_components, x):
    model.n_components = n_components
    x = model.fit_transform(x)
    return x


def main(argc, argv, envp):
    x = make_data()
    print(x)
    model = train_model(x)
    variances = model.explained_variance_
    print(variances)
    threshold = 0.8
    useful_indices = np.where(variances > threshold)[0]
    print(useful_indices)
    n_useful = len(useful_indices)
    print(n_useful)
    x = reduce_model(model, n_useful, x)
    print(x)
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
