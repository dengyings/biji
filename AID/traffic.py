import os
import sys
import numpy as np
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm


class DigitEncoder():

    def fit_transform(self, y):
        return y.astype(int)

    def transform(self, y):
        return y.astype(int)

    def inverse_transform(self, y):
        return y.astype(str)


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(line[:-1].split(','))
    data = np.array(data).T
    encoders, x = [], []
    for row in range(len(data)):
        if data[row, 0].isdigit():
            encoder = DigitEncoder()
        else:
            encoder = sp.LabelEncoder()
        if row < len(data) - 1:
            x.append(encoder.fit_transform(data[row]))
        else:
            y = encoder.fit_transform(data[row])
        encoders.append(encoder)
    x = np.array(x).T
    return encoders, x, y


def train_model(x, y):
    model = svm.SVR(kernel='rbf', C=10, epsilon=0.2)
    model.fit(x, y)
    return model


def pred_model(model, x):
    y = model.predict(x)
    return y


def eval_model(y, pred_y):
    mae = sm.mean_absolute_error(y, pred_y)
    mse = sm.mean_squared_error(y, pred_y)
    mde = sm.median_absolute_error(y, pred_y)
    evs = sm.explained_variance_score(y, pred_y)
    r2s = sm.r2_score(y, pred_y)
    print(mae, mse, mde, evs, r2s)


def make_data(encoders):
    data = [['Tuesday', '13:35', 'San Francisco', 'yes']]
    data = np.array(data).T
    x = []
    for row in range(len(data)):
        encoder = encoders[row]
        x.append(encoder.transform(data[row]))
    x = np.array(x).T
    return x


def main(argc, argv, envp):
    encoders, x, y = read_data('traffic.txt')
    train_x, test_x, train_y, test_y = ms.train_test_split(
        x, y, test_size=0.25, random_state=5)
    model = train_model(train_x, train_y)
    pred_test_y = pred_model(model, test_x)
    eval_model(test_y, pred_test_y)
    x = make_data(encoders)
    pred_y = pred_model(model, x)
    print(encoders[-1].inverse_transform(pred_y))
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
