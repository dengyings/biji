import os
import sys
import platform
import numpy as np
import sklearn.cluster as sc
import sklearn.neighbors as nb
import matplotlib.pyplot as mp


def make_data(a_noise=0.05, n_samples=500):
    t = 2.5 * np.pi * (1 + 2 * np.random.rand(n_samples, 1))
    x = 0.05 * t * np.cos(t)
    y = 0.05 * t * np.sin(t)
    n = a_noise * np.random.rand(n_samples, 2)
    return np.hstack((x, y)) + n


def train_model_no(x):
    model = sc.AgglomerativeClustering(linkage='average',
                                       n_clusters=3
                                       )
    return model


def train_model_10(x):
    model = sc.AgglomerativeClustering(linkage='average',
                                       n_clusters=3, connectivity=nb.kneighbors_graph(x,
                                                                                      10, include_self=False))
    return model


def pred_model(model, x):
    y = model.fit_predict(x)
    return y


def init_model_no():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(121)
    mp.title('Connectivity: no', fontsize=20)
    mp.xlabel('x', fontsize=14)
    mp.ylabel('y', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.1))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def init_model_10():
    mp.subplot(122)
    mp.title('Connectivity: 10', fontsize=20)
    mp.xlabel('x', fontsize=14)
    mp.ylabel('y', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.1))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(x, y):
    mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg', s=80)


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()


def main(argc, argv, envp):
    x = make_data()
    model_no = train_model_no(x)
    model_10 = train_model_10(x)
    pred_no = pred_model(model_no, x)
    pred_10 = pred_model(model_10, x)
    init_model_no()
    draw_chart(x, pred_no)
    init_model_10()
    draw_chart(x, pred_10)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
