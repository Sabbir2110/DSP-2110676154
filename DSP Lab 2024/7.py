import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Part 1: FIR Filter
# y[n] = x[n] + x[n-3]
# -----------------------------

# Input signal
n = np.arange(0, 50)
x = np.sin(2*np.pi*0.1*n)

# FIR filter implementation
y_fir = np.zeros_like(x)
for i in range(len(x)):
    if i >= 3:
        y_fir[i] = x[i] + x[i-3]
    else:
        y_fir[i] = x[i]

# Frequency response
h = np.array([1, 0, 0, 1])  # impulse response
H = np.fft.fft(h, 512)
freq = np.linspace(0, 1, 512)

plt.figure()
plt.plot(freq, np.abs(H))
plt.title("FIR Filter Magnitude Response")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude")
plt.grid()


# -----------------------------
# Part 2: IIR Filter
# Example: y[n] = 0.5x[n] + 0.5y[n-1]
# -----------------------------

y_iir = np.zeros_like(x)

for i in range(len(x)):
    if i == 0:
        y_iir[i] = 0.5 * x[i]
    else:
        y_iir[i] = 0.5 * x[i] + 0.5 * y_iir[i-1]

# Frequency response
w = np.linspace(0, np.pi, 512)
H_iir = 0.5 / (1 - 0.5 * np.exp(-1j*w))

plt.figure()
plt.plot(w/np.pi, np.abs(H_iir))
plt.title("IIR Filter Magnitude Response")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude")
plt.grid()


# -----------------------------
# Part 3: Adaptive Filter (LMS)
# -----------------------------

np.random.seed(0)

# Desired signal (clean)
t = np.arange(0, 200)
d = np.sin(2*np.pi*0.05*t)

# Add noise
noise = 0.5 * np.random.randn(len(t))
x_noisy = d + noise

# LMS parameters
mu = 0.01      # step size
M = 4          # filter length
w = np.zeros(M)

y_lms = np.zeros(len(t))
e = np.zeros(len(t))

# LMS Algorithm
for n in range(M, len(t)):
    x_vec = x_noisy[n:n-M:-1]
    y_lms[n] = np.dot(w, x_vec)
    e[n] = d[n] - y_lms[n]
    w = w + 2 * mu * e[n] * x_vec

# Plot signals
plt.figure()
plt.plot(t, d, label="Original Signal")
plt.plot(t, x_noisy, label="Noisy Signal", linestyle='dashed')
plt.plot(t, y_lms, label="Filtered Signal")
plt.legend()
plt.title("Adaptive Filtering using LMS")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

# Error convergence
plt.figure()
plt.plot(e)
plt.title("Error Converge=-nce (LMS)")
plt.xlabel("Samples")
plt.ylabel("Error")
plt.grid()

plt.show()