import os
import sys
import platform
import numpy as np
import sklearn.pipeline as si
import sklearn.preprocessing as sp
import sklearn.linear_model as sl
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import matplotlib.patches as mc


def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data = [float(substr) for substr in line.split(',')]
            x.append(data[:-1])
            y.append(data[-1])
    return np.array(x), np.array(y)


def train_model(degree, x, y):
    model = si.make_pipeline(sp.PolynomialFeatures(degree),
        sl.LinearRegression())
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


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Polynomial Regression', fontsize=20)
    mp.xlabel('x', fontsize=14)
    mp.ylabel('y', fontsize=14)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_train(train_x, train_y, pred_train_y):
    mp.plot(train_x, train_y, 's', c='limegreen', label='Training')
    sorted_indices = train_x.T[0].argsort()
    mp.plot(train_x.T[0][sorted_indices],
            pred_train_y[sorted_indices], '--', c='dodgerblue',
            label='Predicted Training')
    mp.legend()


def draw_test(test_x, test_y, pred_test_y):
    mp.plot(test_x, test_y, 's', c='orangered', label='Testing')
    mp.plot(test_x, pred_test_y, 'o', c='dodgerblue',
            label='Predicted Testing')
    for x, pred_y, y in zip(test_x, pred_test_y, test_y):
        mp.gca().add_patch(mc.Arrow(
            x, pred_y, 0, y - pred_y, width=0.8, ec='none',
            fc='pink'))
    mp.legend()


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()


def main(argc, argv, envp):
    x, y = read_data('single.txt')
    train_size = int(x.size * 0.8)
    train_x = x[:train_size]
    train_y = y[:train_size]
    model = train_model(3, train_x, train_y)
    pred_train_y = pred_model(model, train_x)
    test_x = x[train_size:]
    test_y = y[train_size:]
    pred_test_y = pred_model(model, test_x)
    eval_model(test_y, pred_test_y)
    init_chart()
    draw_train(train_x, train_y, pred_train_y)
    draw_test(test_x, test_y, pred_test_y)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
