#1. Study of LTI System using Difference Equation and Python Simulation

Consider a discrete-time system defined by the difference equation:
y[n]=0.5x[n]+0.3x[n−1]+0.2x[n−2]
y[n]=0.5x[n]+0.3x[n−1]+0.2x[n−2]

Perform the following tasks:

(i) Determine the impulse response h[n]h[n] of the system.

(ii) Compute the output y[n]y[n] when the input is a unit step signal x[n]=u[n]x[n]=u[n].

Write a Python program to:

    Generate the impulse signal and plot the impulse response

    Generate the unit step signal

    Compute output using convolution (NO FUNCTION USE)

    Plot the output signal

*#

import numpy as np
import matplotlib.pyplot as plt

N = 20
n = np.arange(N)

# -----------------------------
# (i) Impulse Response h[n]
# -----------------------------
impulse = np.zeros(N)
impulse[0] = 1   # δ[n]

h = np.zeros(N)

for i in range(N):
    x0 = impulse[i]
    x1 = impulse[i-1] if i-1 >= 0 else 0
    x2 = impulse[i-2] if i-2 >= 0 else 0
    h[i] = 0.5*x0 + 0.3*x1 + 0.2*x2

# Plot impulse response
plt.figure()
plt.stem(n, h, basefmt=' ')
plt.title("Impulse Response h[n]")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid()

# -----------------------------
# (ii) Output for x[n] = u[n]
# -----------------------------
u = np.ones(N)   # unit step

# Manual convolution
y = np.zeros(N)
for i in range(N):
    for k in range(i+1):
        if k < N and (i-k) < N:
            y[i] += u[k] * h[i-k]

# Plot output
plt.figure()
plt.stem(n, y, basefmt=' ')
plt.title("Output y[n] for x[n]=u[n]")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.grid()

plt.show()
