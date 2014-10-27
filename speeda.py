#!/usr/bin/python

import pylab
import warnings

import numpy
import scipy
import scipy.io.wavfile

def calc_speedup_ratio(audio_file, speed):
    '''Output speed-up ratio of each segment in the audio.'''
    fs, audio = load_audio(audio_file)
    syllable_times = detect_syllables(audio, fs) # list of tuple (start, end)

    # calcDensity() + calcDensityMedian()
#    density = calc_density(syllable_times) # list of density

    # calcSegments() + mergeSegments()
#    segments = calc_segments(density) # list of segment's start point

#    speedup_ratio = calc_speedup_ratio(segments, density, speed) # list of ratio

def detect_syllables(audio, fs):
    nfft = 128
    window = numpy.kaiser(nfft, 0.5)
    S, F, T, _ = pylab.specgram(audio, Fs=fs, window=window, noverlap=64,
                                NFFT=nfft)
    # TODO

def load_audio(audio_file):
    '''Loads audio from AUDIO_FILE and return it.'''
    # Suppress the warning from scipy loading wav file.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sample_rate, audio = scipy.io.wavfile.read(audio_file)
    audio_mono = audio[0, :] # only the first channel is used
    return sample_rate, audio_mono

def gen_audio_segments(segments):
    pass

def gen_render_file(segments):
    pass

def render():
    pass

if __name__ == '__main__':
    segments = calc_speedup_ratio('samples/music.wav', 2)
    gen_audio_segments(segments)
    gen_render_file(segments)
    render()
