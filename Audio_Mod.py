import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from IPython.display import Audio as display
import sys

#message signal

def upload():
    s = input("Enter file path:")
    start_time = int(input("Enter start time:"))
    duration = int(input("Enter duration:"))
    filename = s
    fs, data = wav.read(filename)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    start_sample = int(start_time * fs)
    end_sample = int((start_time + duration) * fs)
    segment_data = data[start_sample:end_sample]
    normalized_data = segment_data / np.max(np.abs(segment_data))
    
    fft_result = np.fft.fft(normalized_data)
    freqs = np.fft.fftfreq(len(fft_result), 1/fs)
    magnitudes = np.abs(fft_result)
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitudes = magnitudes[:len(magnitudes)//2]
    max_freq = np.argmax(positive_freqs)
    
    return normalized_data, fs, max_freq

def sinu(n, fm, duration, sampling_rate, mag=1.0):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    if(n==2):
        y = mag*np.sin(2*np.pi*fm*t)
    else:
        y = mag*np.cos(2*np.pi*fm*t)
    return y

#carrier generation

def generate_carrier(sampling_rate, length, fc, mag=1.0):
    t = np.arange(length) / sampling_rate
    return mag*np.cos(2 * np.pi * fc * t)

#Amplitude modulation

def modulate_dsb_sc(data, fs, carrier_freq, mag=1.0):
    carrier = generate_carrier(fs, len(data), carrier_freq, mag)
    return data * carrier

def modulate_ssb(data, fs, carrier_freq, mag=1.0):
    hilbert_data = signal.hilbert(data)
    carrier = generate_carrier(fs, len(data), carrier_freq, mag)
    ssb_modulated = np.real(hilbert_data) * carrier - np.imag(hilbert_data) * np.sin(2 * np.pi * carrier_freq * np.arange(len(data)) / fs)
    return ssb_modulated

def modulate_am(data, fs, carrier_freq, mag=1.0):
    carrier = generate_carrier(fs, len(data), carrier_freq, mag)
    return (1 + data) * carrier

#Amplitude demodulation

def demodulate_coherent(modulated, fs, sr, carrier_freq, mag=1.0):
    carrier = generate_carrier(sr, len(modulated), carrier_freq, mag)
    de = carrier*modulated
    defft = np.fft.fft(de)
    freq = np.fft.fftfreq(len(de),1/sr)
    defft[np.abs(freq)>fs]=0
    de = np.fft.ifft(defft).real
    return de

#Frequency modulation

def frequency_modulation(signal, fs, fc, kd, mag=1.0):
    time = np.linspace(0, len(signal) / fs, num=len(signal))
    integrated_signal = np.cumsum(signal) / fs
    t = np.arange(len(signal)) / fs
    fm_signal = mag*np.cos(2*np.pi*fc*t + 2*np.pi*kd*integrated_signal)
    return fm_signal

def phase_modulation(signal, fs, fc, kp, mag=1.0):
    time = np.linspace(0, len(signal) / fs, num=len(signal))
    t = np.arange(len(signal)) / fs
    fm_signal = mag*np.cos(2*np.pi*fc*t + kp*signal)
    return fm_signal    

#FM demodulation

def fm_demodulation(fm_signal, fs):
    diff_fm_signal = np.diff(fm_signal) * fs
    envelope = np.abs(signal.hilbert(diff_fm_signal))
    recovered_signal = envelope - np.mean(envelope)
    return recovered_signal

#Plotting signal

def plot_spectrum(signal, fs, title):
    N = len(signal)
    freq = np.linspace(-fs/2, fs/2, N)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(signal)))

    plt.figure(figsize=(10, 4))
    plt.plot(freq, spectrum)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

def plot_time(signal, fs, title):
    time = np.linspace(0, len(signal) / fs, num=len(signal))
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time, signal, label=title, color='b')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()

def main():
    print("1> Manual upload.\n2> Sine wave.\n3> Cosine wave.")
    n=int(input("Please choose your message signal type:"))
    if (n==1):
        data, sr, fs = upload()
        length = len(data)
    elif ((n==2) or (n==3)):
        mag = float(input("Enter magnitude of message signal:"))
        fs = int(input("Enter frequency of message signal:"))
        duration = int(input("Duration of the signal:"))
        sr = int(input("Enter sampling rate:"))
        data = sinu(n,fs,duration,sr, mag)
        length = len(data)
    else:
        print("Invalid choice!")
        
    magc = float(input("Enter magnitude of carrier wave:"))
    print("(Sampling rate/2) > (carrier freq)")
    carrier_freq = int(input("Enter carrier frequency:"))
    xc = generate_carrier(sr, length, carrier_freq, magc)

    print("Type of modulation:\n\t1> DSB\n\t2> AM\n\t3> SSB\n\t4> Phase Modulation\n\t5> FM")
    mtype = int(input("Enter choice:"))

    if (mtype==1):
        modulated = modulate_dsb_sc(data, sr, carrier_freq, magc)
        title = "DSB-SC"
        demodulated = demodulate_coherent(modulated, fs, sr, carrier_freq, magc)
        
    elif (mtype==2):
        modulated = modulate_am(data, sr, carrier_freq, magc)
        title = "AM"
        demodulated = demodulate_coherent(modulated, fs, sr, carrier_freq, magc)
        
    elif (mtype==3):
        modulated = modulate_ssb(data, sr, carrier_freq)
        title = "SSB"
        demodulated = demodulate_coherent(modulated, fs, sr, carrier_freq, magc)
        
    elif (mtype==4):
        kp = float(input("Enter kp:"))
        modulated = phase_modulation(data, sr, carrier_freq, kp, magc)
        title = "PM"
        demodulated = fm_demodulation(modulated,sr)
        demodulated = np.cumsum(demodulated)/sr
        
    elif (mtype==5):
        kd = float(input("Enter kd (The higher, the better):"))
        modulated = frequency_modulation(data, sr, carrier_freq, kd, magc)
        title = "FM"
        demodulated = fm_demodulation(modulated,sr)
        
    else:
        print("Invalid choice!")
        sys.exit(1)
        
    plot_time(data, sr, "Original Time Domain")
    plot_spectrum(data, sr, "Original Frequency Domain")
    
    plot_time(modulated, sr, "Modulated Time Domain")
    plot_spectrum(modulated,sr , "Modulated Frequency Domain")
    
    plot_time(demodulated, sr, "Demodulated Audio Time Domain")
    plot_spectrum(demodulated, sr, "Demodulated Audio Frequency Domain")
    
if __name__ == "__main__":
    main()
