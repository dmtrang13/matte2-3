
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N = 100
kA = 20.0

#98 tilfeldige punkter + 2 randpunkter
x = np.sort(np.concatenate((
    [0.0],
    np.random.uniform(0, 2, N-2),
    [2.0]
)))

#opprett matrise og høyreside
A = np.zeros((N, N))
b = np.zeros(N)

#randbetingelser
A[0, 0] = 1
b[0] = 10

A[-1, -1] = 1
b[-1] = 0

#FVM
for i in range(1, N-1):

    #w = vest, e = øst for i
    dx_w = x[i] - x[i-1]
    dx_e = x[i+1] - x[i]

    #Fouriers lov: q = -kA * du/dx
    a_w = kA / dx_w
    a_e = kA / dx_e

    #-a_W*u_{i-1} + (a_W+a_E)*u_i - a_E*u_{i+1}=0
    #konduktanskoeffisienter
    A[i, i-1] = -a_w
    A[i, i] = a_w + a_e
    A[i, i+1] = -a_e

#løs systemet
u = np.linalg.solve(A, b)

#analytisk løsning
u_exact = 10 - 5*x

plt.figure(figsize=(8, 5))
plt.plot(x, u_exact, 'k-', label="Analytisk")
plt.plot(x, u, 'ro', markersize=3,
         label="FVM - tilfeldige punkter")

plt.xlabel("x")
plt.ylabel("Temperatur u(x)")
plt.title("1D varmeledning - tilfeldig nett")
plt.legend()
plt.grid()
plt.show()

print("Maksimal feil:", np.max(np.abs(u - u_exact)))