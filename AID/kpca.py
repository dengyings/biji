import os
import sys
import platform
import numpy as np
import sklearn.datasets as sd
import sklearn.decomposition as dc
import matplotlib.pyplot as mp


def make_data():
    np.random.seed(7)
    x, y = sd.make_circles(n_samples=500, factor=0.2, noise=0.04)
    return x, y


def pca(x):
    model = dc.PCA()
    x = model.fit_transform(x)
    return x


def kpca(x):
    model = dc.KernelPCA(kernel='rbf', fit_inverse_transform=True,
                         gamma=10)
    x = model.fit_transform(x)
    return model, x


def ikpca(model, x):
    x = model.inverse_transform(x)
    return x


def init_original():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(221)
    mp.title('Original Samples', fontsize=16)
    mp.xlabel('x', fontsize=12)
    mp.ylabel('y', fontsize=12)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(x, y):
    mp.scatter(x[y == 0][:, 0], x[y == 0][:, 1], c='dodgerblue',
               alpha=0.5, s=80, label='Class 0')
    mp.scatter(x[y == 1][:, 0], x[y == 1][:, 1], c='Orangered',
               alpha=0.5, s=80, label='Class 1')
    mp.legend()


def init_pca():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(222)
    mp.title('PCA Samples', fontsize=16)
    mp.xlabel('x', fontsize=12)
    mp.ylabel('y', fontsize=12)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def init_kpca():
    mp.subplot(223)
    mp.title('KPCA Samples', fontsize=16)
    mp.xlabel('x', fontsize=12)
    mp.ylabel('y', fontsize=12)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def init_ikpca():
    mp.subplot(224)
    mp.title('Inverse KPCA Samples', fontsize=16)
    mp.xlabel('x', fontsize=12)
    mp.ylabel('y', fontsize=12)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(x, y):
    mp.scatter(x[y == 0][:, 0], x[y == 0][:, 1], c='dodgerblue',
               alpha=0.5, s=80, label='Class 0')
    mp.scatter(x[y == 1][:, 0], x[y == 1][:, 1], c='Orangered',
               alpha=0.5, s=80, label='Class 1')
    mp.legend()


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.tight_layout()
    mp.show()


def main(argc, argv, envp):
    x, y = make_data()
    pca_x = pca(x)
    model, kpca_x = kpca(x)
    ikpca_x = ikpca(model, kpca_x)
    init_original()
    draw_chart(x, y)
    init_pca()
    draw_chart(pca_x, y)
    init_kpca()
    draw_chart(kpca_x, y)
    init_ikpca()
    draw_chart(ikpca_x, y)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
