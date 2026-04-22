import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Part A: Adaptive Noise Cancellation (LMS)
# -----------------------------

np.random.seed(0)

N = 500
n = np.arange(N)

# Desired clean signal s(n)
s = np.sin(2*np.pi*0.05*n)

# Noise v(n)
v = 0.5 * np.random.randn(N)

# Correlated reference noise x(n)
x = v + 0.1*np.random.randn(N)

# Observed signal
d = s + v

# LMS parameters
M = 5          # filter length
mu = 0.01      # step size
w = np.zeros(M)

y = np.zeros(N)   # filter output
e = np.zeros(N)   # error signal

# LMS Algorithm
for i in range(M, N):
    x_vec = x[i:i-M:-1]
    y[i] = np.dot(w, x_vec)       # estimated noise
    e[i] = d[i] - y[i]            # error ≈ clean signal
    w = w + 2 * mu * e[i] * x_vec

# Plot signals
plt.figure()
plt.plot(n, d, label="Noisy Signal d(n)")
plt.plot(n, y, label="Estimated Noise y(n)")
plt.plot(n, e, label="Recovered Signal e(n)")
plt.legend()
plt.title("Adaptive Noise Cancellation (LMS)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

# Error convergence
plt.figure()
plt.plot(e)
plt.title("Error Signal (Convergence)")
plt.xlabel("Samples")
plt.ylabel("Error")
plt.grid()


# -----------------------------
# Part B: FIR Filter Analysis
# y[n] = x[n] + 0.5x[n-1] + 0.25x[n-2]
# -----------------------------

# Impulse response
h = np.array([1, 0.5, 0.25])

# Transfer Function:
# H(z) = 1 + 0.5z^-1 + 0.25z^-2
print("\nTransfer Function:")
print("H(z) = 1 + 0.5z^-1 + 0.25z^-2")

# Frequency response
H = np.fft.fft(h, 512)
freq = np.linspace(0, 1, 512)

plt.figure()
plt.plot(freq, np.abs(H))
plt.title("FIR Filter Magnitude Response")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude")
plt.grid()

plt.show()