import os
import tempfile
import numpy as np
import soundfile
from parselmouth import Data, praat, Sound


FMIN = 40.
FMAX = 550.

def psola_td(audio,sample_rate,target_pitch=None,fmin=FMIN,fmax=FMAX):
    with tempfile.TemporaryDirectory() as tmpdir:
        if target_pitch is not None:
            audio = pitch_shift(audio, target_pitch, fmin, fmax, sample_rate, tmpdir)
        return audio
    
def psola_stretcher(audio, sample_rate,  constant_stretch=None, fmin=FMIN, fmax=FMAX):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio = time_stretch(audio, constant_stretch, fmin, fmax, sample_rate, tmpdir)
    return audio

def get_manipulation(audio, fmin, fmax, sample_rate, tmpdir):
    audio_file = os.path.join(tmpdir, 'audio.wav')
    save(audio_file, audio, sample_rate)
    return praat.call(Sound(audio_file), "To Manipulation", 1e-3, fmin, fmax)


def pitch_shift(audio, pitch, fmin, fmax, sample_rate, tmpdir):
    pitch = np.copy(pitch)
    pitch[np.isnan(pitch)] = 0.
    pitch_file = os.path.join(tmpdir, 'pitch.txt')
    write_pitch_tier(pitch_file, pitch, float(len(audio)) / sample_rate)
    pitch_tier = Data.read(pitch_file)
    manipulation = get_manipulation(audio, fmin, fmax, sample_rate, tmpdir)
    praat.call([pitch_tier, manipulation], "Replace pitch tier")
    return praat.call(manipulation, "Get resynthesis (overlap-add)").values[0]

def write_pitch_tier(filename, pitch, duration):
    times = np.linspace(0., duration, len(pitch))
    with open(filename, 'w') as file:
        file.write('File type = "ooTextFile"\nObject class = "PitchTier"\n\n')
        file.write('0\n')
        file.write(str(duration) + '\n')
        file.write(str(np.sum(~np.isnan(pitch))) + '\n')
        for time, value in zip(times, pitch):
            if not np.isnan(value):
                file.write(str(time) + '\n' + str(value) + '\n')
                
def write_duration_tier(filename, times, rates, eps=1e-6):
    with open(filename, 'w') as file:
        file.write(
            'File type = "ooTextFile"\nObject class = "DurationTier"\n\n')
        file.write(
            f'xmin = 0.000000\nxmax = {times[-1]:.6f}\npoints: size = {2 * len(times)}\n')
        file.write('points [1]:\n\tnumber = 0\n\tvalue = 1.000000\n')
        for i, (start, end, rate) in enumerate(zip(times[:-1], times[1:], rates)):
            file.write(f'points [{2 * i + 2}]:\n' +
                       f'\tnumber = {start + eps:.6f}\n' +
                       f'\tvalue = {rate:.6f}\n')
            file.write(f'points [{2 * i + 3}]:\n' +
                       f'\tnumber = {end - eps:.6f}\n' +
                       f'\tvalue = {rate:.6f}\n')
        file.write(f'points [{2 * len(times)}]:\n' +
                   f'\tnumber = {times[-1]:.6f}\n' +
                   '\tvalue = 1.000000\n')
                
def save(file, audio, sample_rate):
    soundfile.write(file, audio, sample_rate)
    
def time_stretch(audio, constant_stretch, fmin, fmax, sample_rate, tmpdir):
    if constant_stretch is not None:
        times = np.array([0., len(audio) / sample_rate])
        rates = np.array([1. / constant_stretch])
    duration_file = os.path.join(tmpdir, 'duration.txt')
    write_duration_tier(duration_file, times, rates)
    duration_tier = Data.read(duration_file)
    manipulation = get_manipulation(audio, fmin, fmax, sample_rate, tmpdir)
    praat.call([duration_tier, manipulation], 'Replace duration tier')
    return praat.call(manipulation, "Get resynthesis (overlap-add)").values[0]