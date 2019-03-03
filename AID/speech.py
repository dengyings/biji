import os
import sys
import warnings
import numpy as np
import scipy.io.wavfile as wf
import python_speech_features as sf
import hmmlearn.hmm as hl


def search_speeches(directory, speeches):
    directory = os.path.normpath(directory)
    for entry in os.listdir(directory):
        label = directory[directory.rfind(os.path.sep) + 1:]
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            search_speeches(path, speeches)
        elif os.path.isfile(path) and path.endswith('.wav'):
            if label not in speeches:
                speeches[label] = []
            speeches[label].append(path)


def read_signals(filename):
    sample_rate, sigs = wf.read(filename)
    return sigs, sample_rate


def calc_features(sigs, sample_rate):
    mfcc = sf.mfcc(sigs, sample_rate)
    return mfcc


def read_data(directory):
    speeches = {}
    search_speeches(directory, speeches)
    x, y = [], []
    for label, filenames in speeches.items():
        mfccs = np.array([])
        for filename in filenames:
            sigs, sample_rate = read_signals(filename)
            mfcc = calc_features(sigs, sample_rate)
            mfccs = mfcc if len(mfccs) == 0 else \
                np.append(mfccs, mfcc, axis=0)
        x.append(mfccs)
        y.append(label)
    return x, y


def train_models(x, y):
    models = {}
    for mfccs, label in zip(x, y):
        model = hl.GaussianHMM(n_components=4,
                               covariance_type='diag', n_iter=1000)
        models[label] = model.fit(mfccs)
    return models


def pred_models(models, x):
    y = []
    for mfccs in x:
        best_score, best_label = None, None
        for label, model in models.items():
            score = model.score(mfccs)
            if best_score is None:
                best_score = score
            if best_label is None:
                best_label = label
            if best_score < score:
                best_score, best_label = score, label
        y.append(best_label)
    return y


def main(argc, argv, envp):
    warnings.filterwarnings('ignore',
                            category=DeprecationWarning)
    np.seterr(all='ignore')
    train_x, train_y = read_data('speeches\\training')
    test_x, test_y = read_data('speeches\\testing')
    models = train_models(train_x, train_y)
    pred_test_y = pred_models(models, test_x)
    for y, pred_y in zip(test_y, pred_test_y):
        print(y, pred_y)
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
