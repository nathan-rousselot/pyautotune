import numpy as np
from librosa import pyin

### Time-event based pitch detection
# 1. Zero-crossing rate
def zero_crossing_rate(signal):
    zcr = 0
    zcr_arr = []
    curr_sign = np.sign(signal[0])
    i = 1
    while i < len(signal):
        if np.sign(signal[i]) != curr_sign:
            zcr += 1
            curr_sign = np.sign(signal[i])
            zcr_arr.append(i)
        i += 1
    return zcr, np.array(zcr_arr)

def zrc_to_hz(zcr, fs, length):
    return zcr*fs/(2*length)

# 2. Peak rate
def peak_rate(signal):
    pr = 0
    pr_arr = []
    i = 1
    while i < len(signal) - 1:
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            pr += 1
            pr_arr.append(i)
        i += 1
    return pr, np.array(pr_arr)

def pr_to_hz(pr, fs, length):
    return pr*fs/(length)

# 3. Slope Event Rate
def slope_event_rate(signal, mode='zcr'):
    if mode != 'zcr' and mode != 'pr':
        raise ValueError('Invalid mode, choose between "zcr" and "pr"')
    if mode == 'zcr':
        return zero_crossing_rate(np.gradient(signal))
    else:
        return peak_rate(np.gradient(signal))
    
    
### Auto-correlation based pitch detection
def auto_correlation(signal):
    ac_s = np.correlate(signal, signal, mode='full')
    ac_s = ac_s[len(ac_s)//2:]
    ac_s = ac_s/np.max(ac_s)
    return ac_s

def ac_s_to_hz(ac_s, fs):
    return fs/(100+np.argmax(ac_s[100:]))



### YIN methods for pitch detection
     
def yin_difference(signal,normalize=True):
    # homemade difference: normalizing the output
    tau = np.arange(0, len(signal))
    diff_arr = []
    i = 0
    while i < len(tau):
        diff = 0
        for j in range(len(signal)-tau[i]):
            diff += (signal[j] - signal[j+tau[i]])**2
        diff_arr.append(diff)
        i += 1
    if normalize:
        return np.array(diff_arr)/np.max(diff_arr)
    return np.array(diff_arr)

def yin_difference_to_hz(yin_diff, fs, tol=1e-2):
    _, minima = peak_rate(-yin_diff)
    i = 0
    while yin_diff[minima[i]] > tol:
        i += 1
        if i >= len(minima):
            raise ValueError('Could not find a YIN minimum')
    return fs/(minima[i])

def yin_difference_cumsum(signal):
    diff_arr = yin_difference(signal,normalize=False)
    diff_arr_p = diff_arr
    tau = 0
    while tau < len(diff_arr):
        if tau == 0:
            diff_arr_p[tau] = 1
        else:
            den_sum = 0
            j = 0
            while j < tau:
                den_sum += diff_arr[j]
                j += 1
            diff_arr_p[tau] = diff_arr[tau]/((1/tau)*den_sum)
        tau += 1
    return diff_arr_p

def yin_difference_cumsum_to_hz(yin_diff, fs):
    _, minima = peak_rate(-yin_diff)
    return fs/(minima[1])

### Apply pitch methods to full signal
def pitch_detection(signal, fs, window_size, method='ac'):
    ## Only supports autocorrelation for now
    pitch = []
    full_history = np.zeros_like(signal)
    for i in range(0, len(signal), window_size):
        if i+window_size > len(signal):
            window = signal[i:]
        else:
            window = signal[i:i+window_size]
        if method == 'ac':
            ac_s = auto_correlation(window)
            hz = ac_s_to_hz(ac_s, fs)
            full_history[i:i+window_size] = ac_s
        if method == 'yin':
            yin_diff = yin_difference(window)
            hz = yin_difference_cumsum_to_hz(yin_diff, fs)
            full_history[i:i+window_size] = yin_diff
        pitch.append(hz)
    return np.array(pitch), full_history