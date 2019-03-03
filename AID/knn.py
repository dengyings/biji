import os
import sys
import platform
import numpy as np
import sklearn.neighbors as sn
import matplotlib.pyplot as mp


def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data = [float(substr) for substr in line.split(',')]
            x.append(data[:-1])
            y.append(data[-1])
    return np.array(x), np.array(y)


def train_model(x, y):
    model = sn.KNeighborsClassifier(n_neighbors=10,
                                    weights='distance')
    model.fit(x, y)
    return model


def pred_model(model, x):
    y = model.predict(x)
    return y


def test_data():
    x = np.array([
        [2.2, 6.2],
        [3.6, 1.8],
        [4.5, 3.6]])
    return x


def get_nn(model, x):
    nn_distances, nn_indices = model.kneighbors(x)
    return nn_distances, nn_indices


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('KNN Classifier', fontsize=20)
    mp.xlabel('x', fontsize=14)
    mp.ylabel('y', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)


def draw_grid(grid_x, grid_y):
    mp.pcolormesh(grid_x[0], grid_x[1], grid_y, cmap='brg')
    mp.xlim(grid_x[0].min(), grid_x[0].max())
    mp.ylim(grid_x[1].min(), grid_x[1].max())


def draw_data(train_x, train_y, test_x, pred_test_y, nn_indices):
    classes = np.unique(train_y)
    classes.sort()
    cs = mp.get_cmap('RdYlBu', len(classes))(classes)
    mp.scatter(train_x[:, 0], train_x[:, 1], c=cs[train_y], s=60)
    mp.scatter(test_x[:, 0], test_x[:, 1], marker='D',
               c=cs[pred_test_y], s=120)
    for i, nn_index in enumerate(nn_indices):
        mp.scatter(
            train_x[nn_index, 0], train_x[nn_index, 1],
            marker='D', edgecolor=cs[
                np.ones_like(nn_index) * pred_test_y[i]],
            facecolor='none', linewidth=2, s=180)


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()


def main(argc, argv, envp):
    train_x, train_y = read_data('knn.txt')
    l, r, h = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1, 0.005
    b, t, v = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1, 0.005
    model = train_model(train_x, train_y)
    grid_x = np.meshgrid(np.arange(l, r, h),
                         np.arange(b, t, v))
    grid_y = pred_model(
        model,
        np.c_[grid_x[0].ravel(),
              grid_x[1].ravel()]).reshape(grid_x[0].shape)
    test_x = test_data()
    pred_test_y = pred_model(model, test_x)
    nn_distances, nn_indices = get_nn(model, test_x)
    init_chart()
    draw_grid(grid_x, grid_y)
    draw_data(train_x, train_y.astype(int), test_x,
              pred_test_y.astype(int), nn_indices)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
