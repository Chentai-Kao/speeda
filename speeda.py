#!/usr/bin/python

from lxml.builder import E
import lxml.etree as ET
import os
import subprocess
import warnings

import numpy as np
import pylab
import scipy
import scipy.signal
import scipy.io.wavfile

def calc_speedup_ratio(audio_file, speed):
    '''Output speed-up ratio of each segment in the audio.'''
    audio, fs = load_audio(audio_file)
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
                    np.around(speedup_ratio[i - 1], decimals=2))
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

def load_audio(audio_file):
    # Suppress the warning from scipy loading wav audio_file.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sample_rate, audio = scipy.io.wavfile.read(audio_file)
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

def gen_audio_segments(input_file, segments):
    base_name, extension = os.path.splitext(input_file)
    audio_end = segments[-1].end
    for i in xrange(len(segments)):
        s = segments[i]
        output_file = base_name + '_' + str(i) + extension
        # preserve 1 second at the end of each segment,
        # for MELT (the video editor) to grab frames.
        end = s.end + 1
        if end > audio_end:
            end = audio_end
        # Run 'sox' to generate audio clips.
        command = ['sox', input_file, output_file,\
                   'trim', '%.3f' % s.start, '=%.3f' % end,\
                   'tempo', '-s', '%.2f' % s.ratio]
        subprocess.call(command)

def gen_render_script(video_file, sh_script_path, mlt_script_path, segments):
    # TODO BASH script
    # melt script (XML)
    mlt_script = gen_melt_script(video_file, segments)
    with open(mlt_script_path, 'w') as f:
        f.write("<?xml version='1.0' encoding='utf-8'?>\n")
        f.write(mlt_script)

def gen_melt_script(video_file, segments):
    # Some useful info.
    frame_rate, frame_length = get_video_profiles(video_file)
    video_time = segments[-1].end # video time in second.
    root_dir = os.path.dirname(os.path.abspath(video_file))
    video_base_name = os.path.basename(video_file)
    # XML elements
    mlt = E('mlt', {'title': 'Speeda', 'version': '0.9.0', 'root': root_dir,\
                    'LC_NUMERIC': 'en_US.UTF-8'})
    video_track = E('playlist', {'id': 'playlist1'})
    producer_ids = set() # avoid creating <producer> with same speed-up ratio.
    for s in segments:
        # Producer.
        producer_id = 'slowmotion:2:%.2f' % s.ratio
        producer_frame_length = int(frame_length / s.ratio)
        producer_resource = video_base_name + '?%.2f' % s.ratio
        if producer_id not in producer_ids:
            producer_ids.add(producer_id)
            p = E('producer', {'in': '0',
                               'out': str(producer_frame_length - 1),
                               'id': producer_id})
            p.append(E('property', {'name': 'mlt_type'}, 'producer'))
            p.append(E('property', {'name': 'length'},
                       str(producer_frame_length)))
            p.append(E('property', {'name': 'resource'}, producer_resource))
            p.append(E('property', {'name': 'mlt_service'}, 'framebuffer'))
            mlt.append(p)
        # Clip.
        start_frame = int(s.start / video_time * producer_frame_length)
        end_frame = int(s.end / video_time * producer_frame_length) - 1
        video_track.append(E('entry', {'in': str(start_frame),
                                       'out': str(end_frame),
                                       'producer': producer_id}))
    mlt.append(video_track)
    return ET.tostring(mlt, pretty_print=True)

def get_video_profiles(video_file):
    p = subprocess.Popen(['melt', video_file, '-consumer', 'xml'],
                         stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    # Parse XML.
    root = ET.fromstring(stdout)
    # Get useful info.
    frame_rate = int(root.find('profile').get('frame_rate_num'))
    for property_tag in root.find('producer'):
        if property_tag.get('name') == 'length':
            frame_length = int(property_tag.text)
    return frame_rate, frame_length

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
        return '(start, end, ratio) = (%.2f, %.2f, %.2f)' % (\
                self.start, self.end, self.ratio)

if __name__ == '__main__':
    audio_file = 'playground/short.wav' # TODO extract from video
    video_file = 'playground/short.mp4'
    sh_script_path = 'playground/test.sh'
    mlt_script_path = sh_script_path + '.mlt'

    segments = calc_speedup_ratio(audio_file, 2)
    #gen_audio_segments(audio_file, segments)
    gen_render_script(video_file, sh_script_path, mlt_script_path, segments)
    render()
