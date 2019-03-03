import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wf


def read_tone_freqs(filename):
    with open(filename, 'r') as f:
        tone_freqs = json.loads(f.read())
    print(tone_freqs)
    return tone_freqs


def make_signals(duration, sample_rate, signal_freq, amp):
    times = np.linspace(0, duration, duration * sample_rate)
    sigs = amp * np.sin(2 * np.pi * signal_freq * times)
    return sigs


def synthesizer(tone_freq, tones, sample_rate, amp):
    music_sigs = np.empty(shape=1)
    for tone, duration in tones:
        sigs = make_signals(
            duration, sample_rate, tone_freq[tone], amp)
        music_sigs = np.append(music_sigs, sigs)
    return music_sigs


def save_signals(filename, sample_rate, sigs):
    wf.write(filename, sample_rate,
             (sigs / np.max(sigs) * 2 ** 15).astype(np.int16))


def main(argc, argv, envp):
    tone_freq = read_tone_freqs('equal_tempered_scale.json')
    tones = [
        ('C5', 1.0), ('D5', 0.5), ('E5', 0.5), ('F5', 0.5),
        ('G5', 0.5), ('G5', 0.5), ('F5', 0.5), ('E5', 0.5),
        ('D5', 0.5), ('C5', 1.0)]
    music_sigs = synthesizer(tone_freq, tones, 44100, 1)
    save_signals('syn.wav', 44100, music_sigs)
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv, os.environ))
