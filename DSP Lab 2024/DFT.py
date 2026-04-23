import numpy as np
import matplotlib.pyplot as plt

N = 4
n = np.arange(N)

# Given signal x[n] = sin(nπ/2)
x = np.sin(n*np.pi/2)

# -----------------------------
# (i) Manual DFT
# -----------------------------
X = [0]*N
for k in range(N):
    s = 0
    for i in range(N):
        s += x[i]*np.exp(-2j*np.pi*k*i/N)
    X[k] = s

# -----------------------------
# (ii) NumPy DFT (for verification)
# -----------------------------
X_np = np.fft.fft(x)

# -----------------------------
# Magnitude (NO direct abs use)
# -----------------------------
mag = [np.sqrt(z.real**2 + z.imag**2) for z in X]

# Phase
phase = [np.angle(z) for z in X]

# -----------------------------
# (iv) Manual IDFT
# -----------------------------
x_rec = [0]*N
for i in range(N):
    s = 0
    for k in range(N):
        s += X[k]*np.exp(2j*np.pi*k*i/N)
    x_rec[i] = (s/N).real

# -----------------------------
# Plots
# -----------------------------
k = np.arange(N)

plt.figure()
plt.stem(k, mag, basefmt=' ')
plt.title("Magnitude Spectrum")
plt.xlabel("k")
plt.grid()

plt.figure()
plt.stem(k, phase, basefmt=' ')
plt.title("Phase Spectrum")
plt.xlabel("k")
plt.grid()

plt.figure()
plt.stem(n, x_rec, basefmt=' ')
plt.title("Reconstructed Signal")
plt.xlabel("n")
plt.grid()

plt.show()

# Print
print("x[n]:", x)
print("Manual DFT:", X)
print("NumPy DFT:", X_np)
print("Reconstructed:", np.round(x_rec,3))
