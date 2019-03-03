import os
import sys
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms


def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(line[:-1].split(','))
    data = np.array(data).T
    encoders, x = [], []
    for row in range(len(data)):
        encoder = sp.LabelEncoder()
        if row < len(data) - 1:
            x.append(encoder.fit_transform(data[row]))
        else:
            y = encoder.fit_transform(data[row])
        encoders.append(encoder)
    x = np.array(x).T
    return encoders, x, y


def train_model(x, y):
    model = se.RandomForestClassifier(
        max_depth=8, n_estimators=200, random_state=7)
    model.fit(x, y)
    return model


def eval_cv(model, x, y):
    pc = ms.cross_val_score(model, x, y, cv=2,
                            scoring='precision_weighted')
    rc = ms.cross_val_score(model, x, y, cv=2,
                            scoring='recall_weighted')
    f1 = ms.cross_val_score(model, x, y, cv=2,
                            scoring='f1_weighted')
    ac = ms.cross_val_score(model, x, y, cv=2,
                            scoring='accuracy')
    print('{}% {}% {}% {}%'.format(
          round(pc.mean() * 100, 2), round(rc.mean() * 100, 2),
          round(f1.mean() * 100, 2), round(ac.mean() * 100, 2)))


def make_data(encoders):
    data = [
        ['high', 'med',  '5more', '4', 'big',   'low',  'unacc'],
        ['high', 'high', '4',     '4', 'med',   'med',  'acc'],
        ['low',  'low',  '2',     '4', 'small', 'high', 'good'],
        ['low',  'med',  '3',     '4', 'med',   'high', 'vgood']]
    data = np.array(data).T
    x = []
    for row in range(len(data)):
        encoder = encoders[row]
        if row < len(data) - 1:
            x.append(encoder.transform(data[row]))
        else:
            y = encoder.transform(data[row])
    x = np.array(x).T
    return x, y


def pred_model(model, x):
    y = model.predict(x)
    return y


def eval_ac(y, pred_y):
    ac = ((y == pred_y).sum() / pred_y.size)
    print('Accuracy: {}%'.format(round(ac * 100, 2)))


def main(argc, argv, envp):
    encoders, train_x, train_y = read_data('car.txt')
    model = train_model(train_x, train_y)
    eval_cv(model, train_x, train_y)
    test_x, test_y = make_data(encoders)
    pred_test_y = pred_model(model, test_x)
    eval_ac(test_y, pred_test_y)
    print(encoders[-1].inverse_transform(test_y))
    print(encoders[-1].inverse_transform(pred_test_y))
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
