import os
import sys
import numpy as np
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.naive_bayes as nb


class DigitEncoder():

    def fit_transform(self, y):
        return y.astype(int)

    def transform(self, y):
        return y.astype(int)

    def inverse_transform(self, y):
        return y.astype(str)


def read_data(filename):
    num_less = 0
    num_more = 0
    max_each = 7500
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if '?' not in line:
                line_data = line[:-1].split(', ')
                if line_data[-1] == '<=50K' and \
                        num_less < max_each:
                    data.append(line_data)
                    num_less += 1
                elif line_data[-1] == '>50K' and \
                        num_more < max_each:
                    data.append(line_data)
                    num_more += 1
                if num_less >= max_each and \
                        num_more >= max_each:
                    break
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
    model = nb.GaussianNB()
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


def pred_model(model, x):
    y = model.predict(x)
    return y


def eval_ac(y, pred_y):
    ac = ((y == pred_y).sum() / pred_y.size)
    print('Accuracy: {}%'.format(round(ac * 100, 2)))


def make_data(encoders):
    data = [[
        '39', 'State-gov', '77516', 'Bachelors', '13',
        'Never-married', 'Adm-clerical', 'Not-in-family',
        'White', 'Male', '2174', '0', '40', 'United-States']]
    data = np.array(data).T
    x = []
    for row in range(len(data)):
        encoder = encoders[row]
        x.append(encoder.transform(data[row]))
    x = np.array(x).T
    return x


def main(argc, argv, envp):
    encoders, x, y = read_data('adult.txt')
    train_x, test_x, train_y, test_y = ms.train_test_split(
        x, y, test_size=0.25, random_state=5)
    model = train_model(train_x, train_y)
    eval_cv(model, x, y)
    pred_test_y = pred_model(model, test_x)
    eval_ac(test_y, pred_test_y)
    x = make_data(encoders)
    pred_y = pred_model(model, x)
    print(encoders[-1].inverse_transform(pred_y))
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
