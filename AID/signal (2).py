import os
import sys
import platform
import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp


def show_signal(sigs, sample_rate):
    print(sigs.shape)
    print(sample_rate / 1000, 'kHz')
    print(len(sigs) * 1000 / sample_rate, 'ms')


def read_signals(filename):
    sample_rate, sigs = wf.read(filename)
    show_signal(sigs, sample_rate)
    sigs = sigs[:30] / 2 ** 15
    times = np.arange(30) / sample_rate
    return sigs, sample_rate, times

'''
def init_chart():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('DBSCAN Cluster', fontsize=20)
    mp.xlabel('x', fontsize=14)
    mp.ylabel('y', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator())
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mp.MultipleLocator())
    ax.yaxis.set_minor_locator(mp.MultipleLocator(0.5))
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_chart(x, y, core_mask, offset_mask, periphery_mask):
    labels = set(y)
    cs = mp.get_cmap('brg', len(labels))(range(len(labels)))
    mp.scatter(x[core_mask][:, 0], x[core_mask][:, 1],
               c=cs[y[core_mask]], s=80, label='Core')
    mp.scatter(x[offset_mask][:, 0], x[offset_mask][:, 1],
               marker='x', c=cs[y[offset_mask]], s=80,
               label='Offset')
    mp.scatter(x[periphery_mask][:, 0], x[periphery_mask][:, 1],
               edgecolor=cs[y[periphery_mask]], s=80,
               facecolor='none', label='Periphery')
    mp.legend()


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()
'''


def main(argc, argv, envp):
    sigs, sample_rate, times = read_signals('signal.wav')
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
