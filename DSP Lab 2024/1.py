import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Part 1: Sampling a Cosine Signal
# x(t) = cos(2πf0 t)
# -----------------------------

fs = 1500   # Sampling frequency
t = np.arange(0, 0.01, 1/fs)  # short duration

f0 = 400    # original frequency
x = np.cos(2*np.pi*f0*t)

# Try another frequency that causes aliasing
f_alias = fs - f0   # alias frequency
x_alias = np.cos(2*np.pi*f_alias*t)

# Plot comparison
plt.figure()
plt.stem(t, x, linefmt='-', markerfmt='o', basefmt=' ')
plt.stem(t, x_alias, linefmt='--', markerfmt='x', basefmt=' ')
plt.title("Aliasing: Different Frequencies → Same Samples")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()


# -----------------------------
# Part 2: Original vs Sampled Signal
# -----------------------------

t_cont = np.linspace(0, 0.01, 1000)
x_cont = np.cos(2*np.pi*f0*t_cont)

plt.figure()
plt.plot(t_cont, x_cont, label="Continuous Signal")
plt.stem(t, x, label="Sampled Signal", basefmt=' ')
plt.legend()
plt.title("Continuous vs Sampled Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()


# -----------------------------
# Part 3: Energy vs Power Signal
# x[n] = u[n]
# -----------------------------

N = 100
u = np.ones(N)   # unit step

# Energy calculation
energy = np.sum(np.abs(u)**2)

# Power calculation
power = energy / N

print("Energy of x[n]=u[n]:", energy)
print("Power of x[n]=u[n]:", power)


# -----------------------------
# Part 4: Impulse Representation
# x[n] = sum x[k] δ[n-k]
# -----------------------------

x_seq = np.array([1, 2, 3, 4])

print("\nImpulse Representation:")
for k in range(len(x_seq)):
    print(f"{x_seq[k]} * δ[n-{k}]")

# Optional visualization of impulses
plt.figure()
plt.stem(range(len(x_seq)), x_seq, basefmt=' ')
plt.title("Discrete Signal as Weighted Impulses")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.show()