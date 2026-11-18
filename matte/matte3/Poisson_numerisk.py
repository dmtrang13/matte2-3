
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve

# Rutenett
N = 51
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
h = x[1] - x[0]

X, Y = np.meshgrid(x, y, indexing="ij")

# Gitt virvelstyrke
omega = 2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

# Antall indre punkter
n = N - 2

# 1D Laplace-matrise
T = diags(
    [np.ones(n-1), -2*np.ones(n), np.ones(n-1)],
    [-1, 0, 1],
    format="csr"
)

I = eye(n, format="csr")

# 2D Laplace-operator
A = (kron(I, T) + kron(T, I)) / h**2

# Høyreside, bare indre punkter
b = omega[1:-1, 1:-1].ravel()

# Løs A psi = b
psi_inner = spsolve(A, b)

# Legg inn randverdier psi = 0
psi = np.zeros((N, N))
psi[1:-1, 1:-1] = psi_inner.reshape((n, n))

# Hastighetsfeltet
dpsi_dx, dpsi_dy = np.gradient(psi, h, h)

u = dpsi_dy
v = -dpsi_dx

# Eksakt løsning
psi_exact = (
    -np.sin(np.pi * X) * np.sin(np.pi * Y)
    / np.pi**2
)

# Maksimal feil
error = np.max(np.abs(psi - psi_exact))
print("Maksimal feil:", error)

# Plot strømfunksjon
plt.figure(figsize=(6, 5))
plt.pcolormesh(X, Y, psi, shading="auto", cmap="viridis")
plt.colorbar(label="psi")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Numerisk strømfunksjon")
plt.axis("equal")
plt.show()

# Plot virvelstyrke
plt.figure(figsize=(6, 5))
plt.pcolormesh(X, Y, omega, shading="auto", cmap="RdBu_r")
plt.colorbar(label="omega")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Virvelstyrke")
plt.axis("equal")
plt.show()

# Plot hastighetsfelt
plt.figure(figsize=(6, 5))
plt.quiver(X[::3, ::3], Y[::3, ::3],
           u[::3, ::3], v[::3, ::3])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Hastighetsfelt")
plt.axis("equal")
plt.show()