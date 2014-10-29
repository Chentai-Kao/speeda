#!/usr/bin/python

import os
import subprocess
import warnings

import numpy as np
import pylab
import scipy
import scipy.signal
import scipy.io.wavfile

def calc_speedup_ratio(file_name, speed):
    '''Output speed-up ratio of each segment in the audio.'''
    audio, fs = load_audio(file_name)
    # list of tuple (start, end)
    syllable_times = detect_syllables(audio, fs)
    # calcDensity() + calcDensityMedian() => list of density
    density = calc_density(syllable_times, audio, fs)
    # calcSegments() + mergeSegments() => list of segment's start point
    start_points = calc_segments(density)
    # list of ratio
    speedup_ratio = calc_ratios(start_points, density, speed, audio, fs)
    # Create all segments.
    segments = []
    for i in xrange(1, len(start_points)):
        s = Segment(float(start_points[i - 1]) / 1000,\
                    float(start_points[i]) / 1000,\
                    speedup_ratio[i - 1])
        segments.append(s)
    return segments

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
    density = np.zeros(int(float(audio.size) / fs * 1000), # in ms
                       dtype=np.uint32)
    for start, end in syllable_times:
        vote_start = int(np.floor((start - voteWindow) * 1000)) - 1
        vote_end = int(np.floor((end + voteWindow) * 1000))
        if vote_start < 0:
            vote_start = 0
        if vote_end > density.size:
            vote_end = density.size
        for i in xrange(vote_start, vote_end):
            density[i] += 1
    # Median filtering
    window_size = 151
    return scipy.signal.medfilt(density, window_size)

def calc_segments(density):
    # Calculate the splitting points of segments.
    seg_points = np.array([0], dtype=np.uint32)
    in_valley = False
    valley_start = 0
    for m in xrange(1, density.size):
        if density[m - 1] < density[m]:
            if in_valley: # valley ends
                seg_points = np.append(seg_points, [valley_start, m])
            in_valley = False
        elif density[m - 1] > density[m]:
            valley_start = m
            in_valley = True
    # Make sure 'seg_points' has the end point of 'density'.
    if seg_points[-1] != density.size:
        seg_points = np.append(seg_points, density.size - 1);
    # Merge splitting points to create segment start points.
    min_segment_length = 400 # in ms
    start_points = np.array([0])
    seg_start = seg_points[0]
    for m in xrange(1, seg_points.size):
        if seg_points[m] - seg_start > min_segment_length:
            start_points = np.append(start_points, seg_points[m])
            seg_start = seg_points[m]
    return start_points

def calc_ratios(start_points, density, speed, audio, fs):
    pause_time = 150 # desired pause time (in ms)
    avg_density = np.mean(density)
    # Pause count and speak time.
    pause_count = 0
    speak_time = 0
    for m in xrange(1, start_points.size):
        seg_start, seg_end = start_points[m - 1], start_points[m]
        if is_pause(density[seg_start:seg_end]):
            pause_count += 1
        else:
            ratio = avg_density / np.mean(density[seg_start:seg_end])
            speak_time += float(seg_end - 1 - seg_start) / ratio
    # Calculate desired ratio.
    audio_length = float(audio.size) / fs * 1000 # audio length in ms.
    expected_time = audio_length / speed
    desired_ratio = speak_time / (expected_time - pause_count * pause_time)
    # speed up
    speedup_ratio = np.zeros(0)
    for m in xrange(1, start_points.size):
        seg_start, seg_end = start_points[m - 1], start_points[m]
        if is_pause(density[seg_start:seg_end]):
            ratio = (seg_end - 1 - seg_start) / pause_time
        else:
            ratio = avg_density * desired_ratio /\
                    np.mean(density[seg_start:seg_end])
        speedup_ratio = np.append(speedup_ratio, ratio)
    return speedup_ratio

def is_pause(segment):
    # How much of the segment is zero for it to be considered a pause.
    pause_threshold = 0.5
    return float(np.count_nonzero(segment)) / segment.size < pause_threshold

def load_audio(file_name):
    # Suppress the warning from scipy loading wav file.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sample_rate, audio = scipy.io.wavfile.read(file_name)
    mono = audio[:, 0] # only the first channel is used
    normalized = pcm2float(mono, 'float32')
    return normalized, sample_rate

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

def gen_audio_segments(input_file_name, segments):
    base_name, extension = os.path.splitext(input_file_name)
    audio_end = segments[-1].end
    for i in xrange(len(segments)):
        s = segments[i]
        output_file_name = base_name + '_' + str(i) + extension
        # preserve 1 second at the end of each segment,
        # for MELT (the video editor) to grab frames.
        end = s.end + 1
        if end > audio_end:
            end = audio_end
        # Run 'sox' to generate audio clips.
        command = ['sox', input_file_name, output_file_name,\
                   'trim', '%.3f' % s.start, '=%.3f' % end,\
                   'tempo', '-s', '%.3f' % s.ratio]
        subprocess.call(command)

def gen_render_file(segments):
    pass

def render():
    pass

# A syllable detected by Harma. Only keep relevant info here.
class Syllable:
    def __init__(self, times):
        self.times = times

# A segment with start time, end time, and its speed-up ratio.
# Time is in second. e.g. start = 1.234 means 1.234 second.
class Segment:
    def __init__(self, start, end, ratio):
        self.start = start
        self.end = end
        self.ratio = ratio

    def __str__(self):
        return '(start, end, ratio) = (%.3f, %.3f, %.3f)' % (\
                self.start, self.end, self.ratio)

if __name__ == '__main__':
    file_name = 'playground/music.wav'
    segments = calc_speedup_ratio(file_name, 2)
    gen_audio_segments(file_name, segments)
    gen_render_file(segments)
    render()
