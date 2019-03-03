import os
import sys
import numpy as np
import scipy.io.wavfile as wf
import python_speech_features as sf
import matplotlib.pyplot as mp
import mpl_toolkits.axes_grid1 as mg


def read_signals(filename):
    sample_rate, sigs = wf.read(filename)
    return sigs, sample_rate


def calc_features(sigs, sample_rate):
    mfcc = sf.mfcc(sigs, sample_rate)
    fbnk = sf.logfbank(sigs, sample_rate)
    return mfcc, fbnk


def draw_mfcc(mfcc):
    ma = mp.matshow(mfcc, cmap='gist_rainbow')
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('MFCC', fontsize=20)
    mp.xlabel('Feature', fontsize=14)
    mp.ylabel('Sample', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator(2))
    ax.xaxis.set_minor_locator(mp.MultipleLocator())
    ax.yaxis.set_major_locator(mp.MultipleLocator(2))
    ax.yaxis.set_minor_locator(mp.MultipleLocator())
    mp.tick_params(which='both', top=True, right=True,
                   labeltop=False, labelbottom=True, labelsize=10)
    dv = mg.make_axes_locatable(ax)
    ca = dv.append_axes('right', '3%', pad='3%')
    cb = mp.colorbar(ma, cax=ca)
    cb.set_label('MFCC', fontsize=14)


def draw_fbnk(fbnk):
    ma = mp.matshow(fbnk, cmap='jet')
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.title('Filter Bank', fontsize=20)
    mp.xlabel('Feature', fontsize=14)
    mp.ylabel('Sample', fontsize=14)
    ax = mp.gca()
    ax.xaxis.set_major_locator(mp.MultipleLocator(2))
    ax.xaxis.set_minor_locator(mp.MultipleLocator())
    ax.yaxis.set_major_locator(mp.MultipleLocator(2))
    ax.yaxis.set_minor_locator(mp.MultipleLocator())
    mp.tick_params(which='both', top=True, right=True,
                   labeltop=False, labelbottom=True, labelsize=10)
    dv = mg.make_axes_locatable(ax)
    ca = dv.append_axes('right', '3%', pad='3%')
    cb = mp.colorbar(ma, cax=ca)
    cb.set_label('MFCC', fontsize=14)


def show_chart():
    mp.show()


def main(argc, argv, envp):
    sigs, sample_rate = read_signals('b2.wav')
    mfcc, fbnk = calc_features(sigs, sample_rate)
    print(mfcc.shape, fbnk.shape)
    draw_mfcc(mfcc.T)
    draw_fbnk(fbnk.T)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
