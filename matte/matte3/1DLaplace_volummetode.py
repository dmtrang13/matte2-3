import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)                      # reproducible
N_random = 1000
xs = np.sort(np.random.rand(N_random) * 2.0)  # 1000 points in (0,2)
x = np.concatenate(([0.0], xs, [2.0]))        # include endpoints
x = np.unique(x)                               # ensure unique/sorted
N = len(x)

kA = 20.0
uL, uR = 10.0, 0.0

n = N - 2                       # unknowns are interior nodes 1..N-2
A = np.zeros((n, n))
b = np.zeros(n)

for m, i in enumerate(range(1, N-1)):  # m is eqn index, i is global node
    dx_w = x[i]   - x[i-1]
    dx_e = x[i+1] - x[i]
    a_w  = kA / dx_w
    a_e  = kA / dx_e
    A[m, m] = a_w + a_e

    if i-1 == 0:             # boundary at x=0
        b[m] += a_w * uL
    else:
        A[m, m-1] = -a_w

    if i+1 == N-1:           # boundary at x=2
        b[m] += a_e * uR
    else:
        A[m, m+1] = -a_e

u_int = np.linalg.solve(A, b)

u = np.zeros(N)
u[0], u[-1] = uL, uR
u[1:-1] = u_int

u_exact = 10.0 - 5.0 * x

max_abs_err = np.max(np.abs(u - u_exact))
l2_err = np.sqrt(np.mean((u - u_exact)**2))
print(f"Nodes incl. boundaries: {N}")
print(f"Max abs error: {max_abs_err:.3e}")
print(f"L2 error: {l2_err:.3e}")

plt.figure(figsize=(7, 4.5))
plt.plot(x, u, label="FVM (numerical)")
plt.plot(x, u_exact, "--", label="Analytical: u=10 - 5x")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("1D Laplace (Finite Volume) on Random Nonuniform Mesh")
plt.legend()
plt.tight_layout()
plt.show()
