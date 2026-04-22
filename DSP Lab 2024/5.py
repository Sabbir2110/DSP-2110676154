import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Part 1: DFT and Linearity
# -----------------------------

# Given sequences
x1 = np.array([1, 2, 1, 0])
x2 = np.array([0, 1, 1, 2])

# Compute DFT
X1 = np.fft.fft(x1)
X2 = np.fft.fft(x2)

# Linearity check
y = 3*x1 - 2*x2
Y = np.fft.fft(y)

# RHS of linearity
Y_check = 3*X1 - 2*X2

print("DFT of x1:", X1)
print("DFT of x2:", X2)
print("DFT of y:", Y)
print("3X1 - 2X2:", Y_check)

# Verify (approximately equal)
print("\nLinearity Verified:", np.allclose(Y, Y_check))


# -----------------------------
# Part 2: Spectral Leakage
# -----------------------------

fs = 100            # Sampling frequency
t = np.arange(0, 1, 1/fs)

f = 10.5            # Non-integer frequency (causes leakage)
x = np.sin(2*np.pi*f*t)

# Compute DFT
X = np.fft.fft(x)
freq = np.fft.fftfreq(len(x), d=1/fs)

# Plot spectrum (without window)
plt.figure()
plt.plot(freq[:len(freq)//2], np.abs(X[:len(X)//2]))
plt.title("Spectrum without Window (Spectral Leakage)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()


# -----------------------------
# Part 3: Apply Hamming Window
# -----------------------------

window = np.hamming(len(x))
x_windowed = x * window

# DFT after windowing
Xw = np.fft.fft(x_windowed)

# Plot spectrum (with window)
plt.figure()
plt.plot(freq[:len(freq)//2], np.abs(Xw[:len(Xw)//2]))
plt.title("Spectrum with Hamming Window")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.show()