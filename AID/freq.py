import os
import sys
import platform
import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp


def show_signal(sigs, sample_rate):
    print(sigs.shape)
    print(sample_rate / 1000, 'kHz')
    print(len(sigs) * 1000 / sample_rate, 'ms')


def read_signals(filename):
    sample_rate, sigs = wf.read(filename)
    show_signal(sigs, sample_rate)
    sigs = sigs / 2 ** 15
    times = np.arange(len(sigs)) / sample_rate
    return sigs, sample_rate, times


def time2freq(sigs, sample_rate):
    freqs = nf.fftfreq(len(sigs), d=1 / sample_rate)
    ffts = nf.fft(sigs)
    amps = np.abs(ffts)
    return freqs, ffts, amps


def init_signals():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(121)
    mp.title('Audio Signal', fontsize=20)
    mp.xlabel('Time (ms)', fontsize=14)
    mp.ylabel('Signal', fontsize=14)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_signals(times, sigs):
    times *= 1000
    mp.plot(times, sigs, c='dodgerblue', label='Signal')
    mp.legend()


def init_amplitudes():
    mp.subplot(122)
    mp.title('Audio Amplitude', fontsize=20)
    mp.xlabel('Frequency (kHz)', fontsize=14)
    mp.ylabel('Amplitude', fontsize=14)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_amplitudes(freqs, amps):
    amps = amps[freqs >= 0]
    freqs = freqs[freqs >= 0] / 1000
    mp.semilogy(freqs, amps, c='orangered', label='Amplitude')
    mp.legend()


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.show()


def main(argc, argv, envp):
    sigs, sample_rate, times = read_signals('freq.wav')
    freqs, ffts, amps = time2freq(sigs, sample_rate)
    init_signals()
    draw_signals(times, sigs)
    init_amplitudes()
    draw_amplitudes(freqs, amps)
    show_chart()
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
