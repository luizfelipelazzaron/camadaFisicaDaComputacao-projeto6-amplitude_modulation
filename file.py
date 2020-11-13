import soundfile as sf
import sounddevice as sd
import suaBibSignal as bib
import numpy as np
import matplotlib.pyplot as plt

fs = 44100   # pontos por segundo (frequência de amostragem)
A = 1.5   # Amplitude
F = 1     # Hz
T = 4     # Tempo em que o seno será gerado
t = np.linspace(-T/2, T/2, T*fs)
my_signal = bib.signalMeu()


"""      STEP -> 1 <-        """
my_recording, samplerate = sf.read('camFis.wav')
yAudio = my_recording[:, 1]
samplesAudio = len(yAudio)


# Sem Filtro
X, Y = my_signal.calcFFT(yAudio, samplerate)
plt.figure("Fourier (Sinal de áudio original)")
plt.plot(X, np.abs(Y))
plt.ylabel('Amplitude (m)')
plt.xlabel('Frequencie (Hz)')
plt.grid()
plt.title('Fourier (Sinal de áudio original)')
plt.show()

sd.play(yAudio, fs)
sd.wait()

"""     STEP -> 2 <-        """


def obtaining_norma(data):
    """Return absolute value"""
    maximum = max(data)
    minimum = min(data)
    absolutes = [abs(maximum), abs(minimum)]
    return max(absolutes)


def normalize(data):
    """Recieve data and return normalized data"""
    norma = obtaining_norma(data)
    data = data / norma
    return data


"""     STEP -> 3 <-        """
y_normalized = normalize(yAudio)
# samplesAudio = len(y_normalized)


# Sem Filtro
X, Y = my_signal.calcFFT(y_normalized, samplerate)
plt.figure("Fourier (Sinal de áudio normalizado)")
plt.plot(X, np.abs(Y))
plt.ylabel('Amplitude (m)')
plt.xlabel('Frequencie (Hz)')
plt.grid()
plt.title('Fourier (Sinal de áudio normalizado)')
plt.show()

sd.play(y_normalized, fs)
sd.wait()

"""STEP -> 4 <-"""


def LPF(signal, cutoff_hz, fs):
    from scipy import signal as sg
    #####################
    # Filtro
    #####################
    # https://scipy.github.io/old-wiki/pages/Cookbook/FIRFilter.html
    nyq_rate = fs/2
    width = 5.0/nyq_rate
    ripple_db = 120.0  # dB
    N, beta = sg.kaiserord(ripple_db, width)
    taps = sg.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    return(sg.lfilter(taps, 1.0, signal))


y_filtered = LPF(y_normalized, 4000, fs)

X, Y = my_signal.calcFFT(y_filtered, samplerate)
plt.figure("Fourier (Sinal de áudio filtrado)")
plt.plot(X, Y)
plt.ylabel('Amplitude (m)')
plt.xlabel('Frequencie (Hz)')
plt.grid()
plt.title('Fourier (Sinal de áudio filtrado)')
plt.show()
"""STEP -> 5 <-       """
sd.play(y_filtered, fs)
sd.wait()

""" STEP -> 6 <- """
# Carrier (transportadora)
A_c = 10
f_c = 14000
t = np.linspace(-T/2, T/2, len(y_filtered))
carrier = A_c*np.sin(2*np.pi*f_c*t)

am_signal = [x*y for x, y in zip(y_filtered, carrier)]

X, Y = my_signal.calcFFT(am_signal, samplerate)
plt.figure("Fourier (Sinal de áudio modulado)")
plt.plot(X, Y)
plt.ylabel('Amplitude (m)')
plt.xlabel('Frequencie (Hz)')
plt.grid()
plt.title('Fourier (Sinal de áudio modulado)')
plt.show()

""" STEP -> 7 <- """
y_desmoduled = [x/y for x, y in zip(am_signal, carrier)]

X, Y = my_signal.calcFFT(y_desmoduled, samplerate)
plt.figure("Fourier (Sinal de áudio Desmodularizado)")
plt.plot(X, Y)
plt.ylabel('Amplitude (m)')
plt.xlabel('Frequencie (Hz)')
plt.grid()
plt.title('Fourier (Sinal de áudio Desmodularizado)')
plt.show()

sd.play(y_desmoduled, fs)
sd.wait()
