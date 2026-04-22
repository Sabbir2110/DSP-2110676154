import numpy as np
import matplotlib.pyplot as plt

# Given specifications
fs = 8000        # Sampling frequency
fc = 1000        # Cutoff frequency
N = 21           # Filter length

# Normalized cutoff frequency
fc_norm = fc / fs   # (0 to 0.5)

# Time index centered at zero
n = np.arange(N)
M = (N - 1) / 2

# Ideal impulse response (sinc function)
h_ideal = np.zeros(N)
for i in range(N):
    if n[i] == M:
        h_ideal[i] = 2 * fc_norm
    else:
        h_ideal[i] = np.sin(2 * np.pi * fc_norm * (n[i] - M)) / (np.pi * (n[i] - M))

# Hamming window
w = np.hamming(N)

# Windowed FIR filter
h = h_ideal * w

# -----------------------------
# Plot Impulse Response
# -----------------------------
plt.figure()
plt.stem(n, h, basefmt=' ')
plt.title("Impulse Response (Hamming Window FIR)")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid()

# -----------------------------
# Frequency Response
# -----------------------------
H = np.fft.fft(h, 1024)
freq = np.linspace(0, fs, 1024)

# Magnitude Response
plt.figure()
plt.plot(freq[:512], np.abs(H[:512]))
plt.title("Magnitude Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

# Phase Response
plt.figure()
plt.plot(freq[:512], np.angle(H[:512]))
plt.title("Phase Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()

plt.show()