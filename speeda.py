#!/usr/bin/python

import pylab
import warnings

import numpy as np
import scipy
import scipy.signal
import scipy.io.wavfile

def calc_speedup_ratio(audio_file, speed):
    '''Output speed-up ratio of each segment in the audio.'''
    fs, audio = load_audio(audio_file)
    syllable_times = detect_syllables(audio, fs) # list of tuple (start, end)

    # calcDensity() + calcDensityMedian()
    density = calc_density(syllable_times, audio, fs) # list of density

    # calcSegments() + mergeSegments()
#    segments = calc_segments(density) # list of segment's start point

#    speedup_ratio = calc_speedup_ratio(segments, density, speed) # list of ratio

def detect_syllables(audio, fs):
    syllables = harma(audio, fs)
    syllable_times = []
    for s in syllables:
        start = np.amin(s.times)
        end = np.amax(s.times)
        syllable_times.append((start, end))
    return sorted(syllable_times)

def harma(audio, fs):
    # Parameters for harma.
    nfft = 128
    window = np.kaiser(nfft, 0.5)
    minDB = 20
    # Calculate spectrogram.
    mag, F, T, _ = pylab.specgram(audio, Fs=fs, window=window, noverlap=64,
                                  NFFT=nfft, mode='magnitude')
    # Initialize segmentation parameters.
    syllables = []
    cutoff = None
    # Segment signals into syllables.
    while True:
        # Find the maximum remaining magnitude in the spectrogram.
        freqMax = np.amax(mag, axis=0)
        freqIndex = mag.argmax(axis=0)
        argMax = np.amax(freqMax)
        segmentIndex = freqMax.argmax()
        # Clear temp variables for this iteration.
        times = np.zeros(shape=0)
        segments = np.zeros(shape=0, dtype=np.int)
        freqs = np.zeros(shape=0)
        amps = np.zeros(shape=0)
        # Setup temp variables with initial values.
        segments = np.append(segments, segmentIndex)
        times = np.append(times, T[segmentIndex])
        freqs = np.append(freqs, F[freqIndex[segmentIndex]])
        amps = np.append(amps, 20 * np.log10(argMax))
        # Check if this is the first iteration,
        # if so store the cutoff value for the loop.
        if cutoff is None:
            cutoff = amps[0] - minDB
        # Is it time to stop looking for syllables?
        if amps[0] < cutoff:
            break
        minAmp = amps[0] - minDB
        i = 0
        # Look for all the values less than t with a high enough amplitude.
        t = segmentIndex
        while t > 0 and amps[i] >= minAmp:
            t -= 1
            i += 1
            segments = np.append(segments, t)
            times = np.append(times, T[t])
            freqs = np.append(freqs, F[freqIndex[t]])
            with warnings.catch_warnings(): # suppress divide-by-zero warning
                warnings.simplefilter('ignore')
                amps = np.append(amps, 20 * np.log10(freqMax[t]))
        # Remove the last index because it did not meet criteria.
        if i > 0:
            segments = np.delete(segments, i)
            times = np.delete(times, i)
            freqs = np.delete(freqs, i)
            amps = np.delete(amps, i)
            i -= 1
        # Look for all the values less than t with a high enough amplitude.
        while t < freqIndex.size - 1 and amps[i] >= minAmp:
            t += 1
            i += 1
            segments = np.append(segments, t)
            times = np.append(times, T[t])
            freqs = np.append(freqs, F[freqIndex[t]])
            with warnings.catch_warnings(): # suppress divide-by-zero warning
                warnings.simplefilter('ignore')
                amps = np.append(amps, 20 * np.log10(freqMax[t]))
        # Remove the last index because it did not meet criteria.
        if i > 0:
            segments = np.delete(segments, i)
            times = np.delete(times, i)
            freqs = np.delete(freqs, i)
            amps = np.delete(amps, i)
            i -= 1
        # Store syllable parameters in struct. (irrelevant things are ignored)
        syllable = Syllable(times)
        syllables.append(syllable)
        # Clear the magnitudes for this syllable so that it is not found again.
        mag[:, segments] = 0
    return syllables

def calc_density(syllable_times, audio, fs):
    # Compute density by voting.
    voteWindow = 0.3 # in second
    density = np.zeros(int(np.floor(float(audio.size) / fs * 1000))) # len in ms
    for start, end in syllable_times:
        vote_start = int(np.floor((start - voteWindow) * 1000))
        vote_end = int(np.floor((end + voteWindow) * 1000))
        if vote_start < 0:
            vote_start = 0
        if vote_end >= density.size:
            vote_end = density.size - 1
        for i in xrange(vote_start, vote_end + 1):
            density[i] += 1
    # Median filtering
    window_size = 151
    scipy.signal.medfilt(density, window_size)
    return density

def load_audio(audio_file):
    '''Loads audio from AUDIO_FILE and return it.'''
    # Suppress the warning from scipy loading wav file.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sample_rate, audio = scipy.io.wavfile.read(audio_file)
    mono = audio[:, 0] # only the first channel is used
    normalized = pcm2float(mono, 'float32')
    return sample_rate, normalized

def pcm2float(sig, dtype='float64'):
    """Excerpted from mgeier on Github."""
    sig = np.asarray(sig)
    if sig.dtype.kind != 'i':
        raise TypeError("'sig' must be an array of signed integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be floating point type")

    # Note that 'min' has a greater (by 1) absolute value than 'max'!
    # Therefore, we use '-min' here to avoid clipping.
    return sig.astype(dtype) / dtype.type(-np.iinfo(sig.dtype).min)

def gen_audio_segments(segments):
    pass

def gen_render_file(segments):
    pass

def render():
    pass

# A syllable detected by Harma. Only keep relevant info here.
class Syllable:
    def __init__(self, times):
        self.times = times

if __name__ == '__main__':
    segments = calc_speedup_ratio('samples/music.wav', 2)
    gen_audio_segments(segments)
    gen_render_file(segments)
    render()
