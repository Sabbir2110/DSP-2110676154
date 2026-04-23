#  2. Computation of Discrete Fourier Transform and its Analysis using Python

Given Signal:
x[n]=[1,2,3,4],N=4
x[n]=[1,2,3,4],N=4

Perform the following tasks:

(i) Compute the DFT of the signal

(ii) Compute the DFT using Python (NumPy)

(iii) Find and plot:

    Magnitude spectrum ∣X[k]∣∣X[k]∣

    Phase spectrum ∠X[k]∠X[k]

(iv) Compute the Inverse DFT (IDFT) and reconstruct the original signal

Write a Python program to:

    Magnitude Spectrum Plot (NO FUNCTION USE)

    Phase Spectrum Plot

    Plot the Reconstructed signal (NO FUNCTION USE)

*#

import numpy as np
import matplotlib.pyplot as plt

# Given signal
x = [1,2,3,4]
N = 4

# -----------------------------
# (i) Manual DFT
# -----------------------------
X = [0]*N
for k in range(N):
    s = 0
    for n in range(N):
        s += x[n] * np.exp(-2j*np.pi*k*n/N)
    X[k] = s

# -----------------------------
# Magnitude (NO function use)
# -----------------------------
mag = [np.sqrt((z.real)**2 + (z.imag)**2) for z in X]

# Phase
phase = [np.angle(z) for z in X]

# -----------------------------
# (iv) Manual IDFT
# -----------------------------
x_rec = [0]*N
for n in range(N):
    s = 0
    for k in range(N):
        s += X[k] * np.exp(2j*np.pi*k*n/N)
    x_rec[n] = (s/N).real

# -----------------------------
# Plots
# -----------------------------
k = np.arange(N)

plt.figure()
plt.stem(k, mag, basefmt=' ')
plt.title("Magnitude Spectrum |X[k]|")
plt.xlabel("k")
plt.grid()

plt.figure()
plt.stem(k, phase, basefmt=' ')
plt.title("Phase Spectrum")
plt.xlabel("k")
plt.grid()

plt.figure()
plt.stem(range(N), x_rec, basefmt=' ')
plt.title("Reconstructed Signal")
plt.xlabel("n")
plt.grid()

plt.show()

# Print results
print("DFT X[k]:", X)
print("Reconstructed x[n]:", np.round(x_rec,3))
