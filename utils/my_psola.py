### Some implementations of Psola algorithms

from librosa import yin
import numpy as np
import matplotlib.pyplot as plt


def pitch_synchronous_analysis(signal, samplerate, fmin=80, fmax=1000, window_duration=2e-2, mu=2):
    # pitch synchronous analysis, should be single voice, single note
    frame_size = int(window_duration * samplerate) # framsize of 20ms (nearest integer)
    pitch = yin(signal, fmin=fmin, fmax=fmax, sr=samplerate, frame_length=frame_size)
    f0 = np.median(pitch)
    h_length = int((samplerate/f0)*mu)
    h = np.hamming(h_length)
    n = int(len(signal)//(samplerate//f0)) # number of frames
    pitch_synchronous_signal = np.zeros((n, h_length))
    for i in range(n):
        if i*int(samplerate//f0)+h_length > len(signal):
            pitch_synchronous_signal = pitch_synchronous_signal[:i, :]
            break
        pitch_synchronous_signal[i,:] = signal[i*int(samplerate//f0):i*int(samplerate//f0)+h_length]*h
    print("Detected f0: ", f0, "Hz")
    return pitch_synchronous_signal, f0

def display_pitch_synchronous_signal(pitch_synchronous_signal):
    n = pitch_synchronous_signal.shape[0]
    if n > 10:
        n = 10
    plt.figure(figsize=(15,3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.plot(pitch_synchronous_signal[i,:])
    plt.show()
    
def td_psola(signal, samplerate, target_f0=440, mu=2):
    pitch_synchronous_signal, f0 = pitch_synchronous_analysis(signal, samplerate, mu=mu)
    speed_factor = target_f0/f0
    n = pitch_synchronous_signal.shape[0]
    h_length = pitch_synchronous_signal.shape[1]
    signal_length = int(n*(samplerate/f0))
    signal = np.zeros(int((signal_length+h_length//2)*1/speed_factor))
    for i in range(n):
        if i == n-1:
            break
            #signal[int(i*(samplerate/f0)*1/speed_factor):] += pitch_synchronous_signal[i,:] ## NEEDS TO BE FIXED ASAP
        else:
            signal[int(i*(samplerate/f0)*1/speed_factor):int(i*(samplerate/f0)*1/speed_factor)+h_length] += pitch_synchronous_signal[i,:]
    
    ### with no speed factor
    #for i in range(n):
    #    signal[i*int(samplerate//f0):i*int(samplerate//f0)+h_length] += pitch_synchronous_signal[i,:]
    return signal
    
def remove_silence(signal):
    start = 0
    step = 10
    while signal[start:start+step].mean() < 1e-2:
        start += step
    end = len(signal)
    while signal[end-step:end].mean() < 1e-2:
        end -= step
    return np.array(signal[start:end])

def detect_pitch_change(pitch, threshold=1.5e-3):
    pitch = np.gradient(pitch)
    pitch = np.abs(pitch)
    pitch /= np.max(pitch)
    pitch = np.log10(pitch + 1)
    index_of_change = np.where(pitch > threshold)[0]
    i = 0
    while i < len(index_of_change):
        if i+1 == len(index_of_change):
            break
        if index_of_change[i+1] - index_of_change[i] < 10:
            if pitch[index_of_change[i+1]] > pitch[index_of_change[i]]:
                index_of_change = np.delete(index_of_change, i)
            else:
                index_of_change = np.delete(index_of_change, i+1)
        else:
            i += 1
    return index_of_change

def split_signal(signal, index_of_change, pitch, window_size=1000):
    signals = []
    pitches = []
    start = 0
    i_prev = -1
    for i in index_of_change:
        signals.append(signal[start:i*window_size])
        if i_prev == -1:
            pitches.append(np.median(pitch[:i]))
        else:
            pitches.append(np.median(pitch[i_prev:i]))
        start = i*window_size
        i_prev = i 
    signals.append(signal[start:])
    pitches.append(np.median(pitch[i_prev:]))
    return signals, pitches

def recombine_signals(signals):
    signal = np.array([])
    for s in signals:
        signal = np.append(signal, s)
    return signal