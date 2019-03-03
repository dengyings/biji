import os
import sys
import platform
import numpy as np
import sklearn.neighbors as sn
import matplotlib.pyplot as mp
import matplotlib.patches as mc


def make_data():
    x = np.array([
        [3.0, 7.0],
        [2.3, 9.0],
        [1.7, 9.0],
        [1.0, 7.0],
        [1.0, 5.0],
        [1.7, 3.0],
        [3.0, 1.0],
        [4.3, 3.0],
        [5.0, 5.0],
        [5.0, 7.0],
        [4.3, 9.0],
        [3.7, 9.0]])
    return x


def train_model(x):
    model = sn.NearestNeighbors(n_neighbors=3,
                                algorithm='ball_tree')
    model.fit(x)
    return model


def test_data():
    x = np.array([
        [2.3, 8.3],
        [1.7, 7.5],
        [1.7, 6.5],
        [2.3, 5.0],
        [2.8, 4.7],
        [3.0, 5.5],
        [3.2, 4.7],
        [3.7, 5.0],
        [4.3, 7.5],
        [4.3, 6.5],
        [3.7, 8.3],
        [2.3, 4.0],
        [3.0, 2.0],
        [3.7, 4.0]])
    return x


def pred_model(model, x):
    nn_distance, nn_indices = model.kneighbors(x)
    return nn_distance, nn_indices


def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Find Nearest Neighbors', fontsize=20)
    mp.xlabel('x', fontsize=14)
    mp.ylabel('y', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    mp.tick_params(which='both', top=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_train(train_x):
    mp.scatter(train_x[:, 0], train_x[:, 1], c='k', zorder=2)


def draw_test(test_x, train_x, nn_indices):
    cs = mp.get_cmap('gist_rainbow', len(nn_indices))(
        range(len(nn_indices)))
    for i, nn_index in enumerate(nn_indices):
        mp.gca().add_patch(mc.Polygon(
            train_x[nn_index], ec='none', fc=cs[i], alpha=0.2,
            zorder=0))
        mp.scatter(test_x[i, 0], test_x[i, 1], c=cs[i], s=80,
                   zorder=1)


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()


def main(argc, argv, envp):
    train_x = make_data()
    model = train_model(train_x)
    test_x = test_data()
    nn_distance, nn_indices = pred_model(model, test_x)
    init_chart()
    draw_train(train_x)
    draw_test(test_x, train_x, nn_indices)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
