from my_psola import *
from pitch_detectors import *
from signal_manager import *
from td_psola import *
from librosa import pyin
from scipy.io import wavfile
from IPython.display import Audio

FMIN = 80
FMAX = 1000

def shift_full_pitch(signal, samplerate, shift, fmin=FMIN, fmax=FMAX, pitch=None):
    if pitch is None:
        pitch, _, _ = pyin(signal, fmin=fmin, fmax=fmax, sr=samplerate, frame_length=int(2e-2*samplerate))
    return psola_td(signal, samplerate, pitch+shift, fmin, fmax)

def change_pitch(signal, samplerate, target_pitch, fmin=FMIN, fmax=FMAX, pitch=None):
    if pitch is None:
        pitch, _, _ = pyin(signal, fmin=fmin, fmax=fmax, sr=samplerate, frame_length=int(2e-2*samplerate))
    return psola_td(signal, samplerate, target_pitch, fmin, fmax)

def change_duration(signal, samplerate, constant_stretch, fmin=FMIN, fmax=FMAX):
    return psola_stretcher(signal, samplerate, constant_stretch, fmin, fmax)
    