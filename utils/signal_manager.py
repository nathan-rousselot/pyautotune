import numpy as np
import matplotlib.pyplot as plt


def generate_signal(f0=440, fs=44100, number_of_periods=1, duration=None):
    if duration is None:
        t = np.arange(0, number_of_periods/f0*fs)
    else:
        t = np.arange(0, duration*fs)
    return np.sin(2*np.pi*f0*t/fs)

def generate_pseudoperiodic(f0=440, fs=44100, number_of_periods=1, duration=None):
    if duration is None:
        t = np.arange(0, number_of_periods/f0*fs)
    else:
        t = np.arange(0, duration*fs)
    return np.sin(2*np.pi*f0*t/fs) + np.sin(2*np.pi*2*f0*t/fs) + np.sin(2*np.pi*3*f0*t/fs) + np.random.randn(np.size(t))
    
def compute_cepstrum(signal, fs):
    return np.fft.ifft(np.log(np.abs(np.fft.fft(signal))))

def noise_signal(signal,sigma=0.1):
    return signal + np.random.normal(0, sigma, len(signal))

def visualize_signal(signal, fs, zrc_arr=None, pr_arr=None, zrc_slope_arr=None, pr_slope_arr=None, ac_s=None, yin_diff=None, cepstrum=None):
    time = np.arange(0, len(signal)/fs, 1/fs)
    if zrc_arr is None and pr_arr is None and zrc_slope_arr is None and pr_slope_arr is None and ac_s is None and yin_diff is None and cepstrum is None:
        signal_gradient = np.gradient(signal)
        plt.figure(figsize=(10,3))
        plt.subplot(1, 2, 1)
        plt.plot(time,signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal')
        plt.subplot(1, 2, 2)
        plt.plot(time,signal_gradient)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal Gradient')
        plt.show()
    if zrc_arr is not None:
        plt.figure()
        plt.plot(time,signal,label="Signal")
        plt.plot(zrc_arr/fs, signal[zrc_arr], 'ro', label="Zero Crossing")
        plt.legend(loc="upper right")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Zero Crossing Detection')
        plt.show()
    if pr_arr is not None:
        plt.figure()
        plt.plot(time,signal,label="Signal")
        plt.plot(pr_arr/fs, signal[pr_arr], 'ro', label="Peak Detection")
        plt.legend(loc="lower right")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Peak Detection')
        plt.show()
    if zrc_slope_arr is not None:
        plt.figure(figsize=(10,3))
        plt.subplot(1, 2, 1)
        plt.plot(time,signal,label="Signal")
        plt.plot(zrc_slope_arr/fs, signal[zrc_slope_arr], 'ro', label="Zero Crossing Slope")
        plt.legend(loc="upper right")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal')
        plt.subplot(1, 2, 2)
        plt.plot(time,np.gradient(signal),label='Signal Gradient')
        plt.plot(zrc_slope_arr/fs, np.gradient(signal)[zrc_slope_arr], 'ro', label="Zero Crossing Slope")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal Gradient')
        plt.legend(loc="upper right")
        plt.show()
    if pr_slope_arr is not None:
        plt.figure(figsize=(10,3))
        plt.subplot(1,2,1)
        plt.plot(time,signal,label="Signal")
        plt.plot(pr_slope_arr/fs, signal[pr_slope_arr], 'ro', label="Peak Slope Detection")
        plt.legend(loc="upper right")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal')
        plt.subplot(1,2,2)
        plt.plot(time,np.gradient(signal),label='Signal Gradient')
        plt.plot(pr_slope_arr/fs, np.gradient(signal)[pr_slope_arr], 'ro', label="Peak Slope Detection")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal Gradient')
        plt.legend(loc="upper right")
        plt.show()
    if ac_s is not None:
        plt.figure()
        plt.plot(time, signal, label='Signal')
        plt.plot(time, ac_s, label='Auto-correlation')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Auto-correlation vs Signal')
        plt.legend(loc="upper right")
        plt.show()
    if yin_diff is not None:
        plt.figure()
        plt.plot(time, signal, label='Signal')
        plt.plot(time, yin_diff, label='YIN difference function')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('YIN Difference vs Signal')
        plt.legend(loc="upper right")
        plt.show()
    if cepstrum is not None:
        plt.figure(figsize=(10,3))
        plt.subplot(1, 2, 1)
        plt.plot(time,signal,label="Signal")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal')
        plt.subplot(1, 2, 2)
        plt.plot(time,cepstrum,label='Cepstrum')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Cepstrum')
        plt.show()