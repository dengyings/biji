import os
import sys
import platform
import numpy as np
import sklearn.decomposition as dc
import matplotlib.pyplot as mp


def read_data(filename):
    x = np.loadtxt(filename)
    return x


def ica(x):
    model = dc.FastICA(n_components=x.shape[1])
    x = model.fit_transform(x)
    return x


def init_original():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(211)
    mp.title('Original', fontsize=16)
    mp.xlabel('Time', fontsize=12)
    mp.ylabel('Signal', fontsize=12)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def init_ica():
    mp.subplot(212)
    mp.title('ICA', fontsize=16)
    mp.xlabel('Time', fontsize=12)
    mp.ylabel('Signal', fontsize=12)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(x):
    x = x.T
    for i, component in enumerate(x):
        mp.plot(component, label='Component %d' % (i + 1))
    #mp.plot(x.sum(axis=0), c='black', label='Mixture')
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
    x = read_data('signals.txt')
    ica_x = ica(x)
    init_original()
    draw_chart(x)
    init_ica()
    draw_chart(ica_x)
    show_chart()
    return 0

if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
