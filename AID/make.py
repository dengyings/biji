import os
import sys
import platform
import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp


def nosied_signals(duration, sample_rate, signal_freq):
    times = np.linspace(0, duration, duration * sample_rate)
    nosied_sigs = np.sin(2 * np.pi * signal_freq * times) + \
        np.random.randn(times.size) / 2
    return times, nosied_sigs


def time2freq(sigs, sample_rate):
    freqs = nf.fftfreq(sigs.size, d=1 / sample_rate)
    ffts = nf.fft(sigs)
    print(ffts)
    amps = np.abs(ffts)
    return freqs, ffts, amps


def filter(freqs, noised_ffts, noised_amps):
    fund_freq = np.abs(freqs[noised_amps.argmax()])
    hight_freqs = np.where(np.abs(freqs) != fund_freq)
    filtered_ffts = noised_ffts.copy()
    filtered_ffts[hight_freqs] = 0
    filtered_amps = noised_amps.copy()
    filtered_amps[hight_freqs] = 0
    return fund_freq, filtered_ffts, filtered_amps


def freq2time(ffts):
    sigs = nf.ifft(ffts)
    print(sigs)
    return sigs.real


def save_signals(filename, sample_rate, sigs):
    wf.write(filename, sample_rate,
             (sigs / np.max(sigs) * 2 ** 15).astype(np.int16))


def init_noised_signals():
    mp.gcf().set_facecolor(np.ones(3) * 240 / 255)
    mp.subplot(221)
    mp.title('Noised Signal', fontsize=16)
    mp.xlabel('Time (ms)', fontsize=12)
    mp.ylabel('Signal', fontsize=12)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_noised_signals(times, noised_sigs):
    times, noised_sigs = times[:200] * 1000, noised_sigs[:200]
    mp.plot(times, noised_sigs, c='orangered', label='Nosied',
            zorder=0)
    mp.scatter(times, noised_sigs, edgecolor='orangered',
               facecolor='white', label='Sample', zorder=1)
    mp.legend()


def init_noised_amplitudes():
    mp.subplot(222)
    mp.title('Noised Amplitude', fontsize=15)
    mp.xlabel('Frequency (kHz)', fontsize=12)
    mp.ylabel('Amplitude', fontsize=12)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_noised_amplitudes(freqs, noised_amps):
    noised_amps = noised_amps[freqs >= 0]
    freqs = freqs[freqs >= 0] / 1000
    mp.semilogy(freqs, noised_amps, c='limegreen',
                label='Noised')
    mp.legend()


def init_filtered_amplitudes():
    mp.subplot(224)
    mp.title('Filtered Amplitude', fontsize=15)
    mp.xlabel('Frequency (kHz)', fontsize=12)
    mp.ylabel('Amplitude', fontsize=12)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_filtered_amplitudes(freqs, filtered_amps):
    filtered_amps = filtered_amps[freqs >= 0] + 1
    freqs = freqs[freqs >= 0] / 1000
    mp.semilogy(freqs, filtered_amps, c='dodgerblue',
                label='Filtered')
    mp.legend()


def init_filtered_signals():
    mp.subplot(223)
    mp.title('Filtered Signal', fontsize=16)
    mp.xlabel('Time (ms)', fontsize=12)
    mp.ylabel('Signal', fontsize=12)
    mp.tick_params(which='both', top=True, right=True,
                   labelright=True, labelsize=10)
    mp.grid(linestyle=':')


def draw_filtered_signals(times, filtered_sigs):
    times, filtered_sigs = times[:200] * 1000, filtered_sigs[:200]
    mp.plot(times, filtered_sigs, c='hotpink', label='Filtered',
            zorder=0)
    mp.scatter(times, filtered_sigs, edgecolor='hotpink',
               facecolor='white', label='Sample', zorder=1)
    mp.legend()


def show_chart():
    mng = mp.get_current_fig_manager()
    if 'Windows' in platform.system():
        mng.window.state('zoomed')
    else:
        mng.resize(*mng.window.maxsize())
    mp.tight_layout()
    mp.show()
'''
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
'''


def main(argc, argv, envp):
    duration = 5
    sample_rate = 44100
    signal_freq = 750
    times, noised_sigs = nosied_signals(
        duration, sample_rate, signal_freq)
    freqs, noised_ffts, noised_amps = time2freq(
        noised_sigs, sample_rate)
    fund_freq, filtered_ffts, filtered_amps = filter(
        freqs, noised_ffts, noised_amps)
    print(fund_freq)
    filtered_sigs = freq2time(filtered_ffts)
    save_signals('noised.wav', sample_rate, noised_sigs)
    save_signals('filtered.wav', sample_rate, filtered_sigs)
    init_noised_signals()
    draw_noised_signals(times, noised_sigs)
    init_noised_amplitudes()
    draw_noised_amplitudes(freqs, noised_amps)
    init_filtered_amplitudes()
    draw_filtered_amplitudes(freqs, filtered_amps)
    init_filtered_signals()
    draw_filtered_signals(times, filtered_sigs)
    show_chart()
    '''
    sigs, sample_rate, times = read_signals('freq.wav')
    freqs, ffts, amps = time2freq(sigs, sample_rate)
    init_signals()
    draw_signals(times, sigs)
    init_amplitudes()
    draw_amplitudes(freqs, amps)
    show_chart()
    '''
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
