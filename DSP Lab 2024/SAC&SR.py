import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PART A: SAMPLING + ALIASING + FOLDING FREQUENCY
# =========================================================

Fs = 1500  # sampling frequency

# Continuous-time signal definition (use dense time for visualization)
t = np.linspace(0, 0.01, 5000)

x_t = 5*np.sin(1000*np.pi*t) + 3*np.sin(2000*np.pi*t)

# Sampling
t_s = np.arange(0, 0.01, 1/Fs)
x_n = 5*np.sin(1000*np.pi*t_s) + 3*np.sin(2000*np.pi*t_s)

# Frequencies
f1 = 1000*np.pi / (2*np.pi)   # 500 Hz
f2 = 2000*np.pi / (2*np.pi)   # 1000 Hz

# Nyquist / Folding frequency
f_nyquist = Fs / 2

print("Frequency components:")
print("f1 =", f1, "Hz")
print("f2 =", f2, "Hz")
print("Nyquist (folding frequency) =", f_nyquist, "Hz")

# Aliasing check
def alias(f, Fs):
    return abs(f - round(f/Fs)*Fs)

print("\nAliased frequencies:")
print("f1 alias =", alias(f1, Fs), "Hz")
print("f2 alias =", alias(f2, Fs), "Hz")

# Plot continuous vs sampled
plt.figure()
plt.plot(t, x_t, label="Continuous Signal")
plt.stem(t_s, x_n, linefmt='r-', markerfmt='ro', basefmt=' ')
plt.title("Sampling of Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()


# =========================================================
# PART B: MULTICHANNEL SIGNAL INTERPRETATION
# =========================================================

# Components
x1 = 5*np.sin(2*np.pi*500*t_s)
x2 = 3*np.sin(2*np.pi*1000*t_s)

plt.figure()
plt.plot(t_s, x1, label="500 Hz component")
plt.plot(t_s, x2, label="1000 Hz component")
plt.plot(t_s, x1 + x2, label="Combined Signal", linestyle='dashed')
plt.title("Multichannel Signal Representation")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()


# =========================================================
# PART C: AUTOCORRELATION
# x(n) = [1,2,1,1]
# =========================================================

x = np.array([1, 2, 1, 1])

# autocorrelation
r = np.correlate(x, x, mode='full')

lags = np.arange(-len(x)+1, len(x))

print("\nAutocorrelation values:")
for lag, val in zip(lags, r):
    print(f"lag {lag}: {val}")

# Plot autocorrelation
plt.figure()
plt.stem(lags, r, basefmt=' ')
plt.title("Autocorrelation")
plt.xlabel("Lag")
plt.ylabel("Rxx")
plt.grid()

print("\nMaximum occurs at lag =", lags[np.argmax(r)])


# =========================================================
# PART D: IMPULSE REPRESENTATION + CONVOLUTION
# =========================================================

# Signal representation using impulses
x_imp = np.array([1, 2, 1, 1])
h_imp = np.array([1, 0.5, 0.25])

print("\nImpulse representation:")
for i, val in enumerate(x_imp):
    print(f"{val} δ[n-{i}]")

# Convolution
y = np.convolve(x_imp, h_imp)

print("\nConvolution result:", y)

# Plot convolution output
plt.figure()
plt.stem(y, basefmt=' ')
plt.title("Convolution Output")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.show()