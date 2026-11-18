import numpy as np
import matplotlib.pyplot as plt

L = 2.0
c = 0.25
c2 = c**2

# Grid
Nx = 401
x = np.linspace(0.0, L, Nx)
dx = x[1] - x[0]

# Steady (Laplace) analytic solution expected for these BCs
u_steady = 10.0 + (0.0 - 10.0) * (x / L)  # line from 10 at x=0 to 0 at x=2, i.e., 10 - 5x

# Stability: dt <= dx^2 / (2 * c2)
dt_max = dx**2 / (2.0 * c2)
dt = 0.9 * dt_max  # a bit below the limit for safety

# Time horizon
t_end = 5.0
Nt = int(np.ceil(t_end / dt))

# Initial condition
u = np.zeros_like(x)

# Enforce boundary conditions at t=0
u[0] = 10.0
u[-1] = 0.0

snap_times = [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
snap_indices = set(int(round(t / dt)) for t in snap_times)

snapshots = {}
t = 0.0
alpha = c2 * dt / dx**2  # scheme coefficient

for n in range(Nt + 1):
    if n in snap_indices:
        snapshots[round(t, 4)] = u.copy()
    if n == Nt:
        break
    u_new = u.copy()
    u_new[1:-1] = u[1:-1] + alpha * (u[2:] - 2.0*u[1:-1] + u[:-2])
    u_new[0] = 10.0
    u_new[-1] = 0.0
    u = u_new
    t += dt

plt.figure(figsize=(7, 4.5))
plt.plot(x, u_steady, linestyle='--', label="Steady (Laplace): 10 - 5x")
for ts in snap_times:
    key = round(ts, 4)
    if key in snapshots:
        plt.plot(x, snapshots[key], label=f"t={ts:.1f}s")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title(f"1D Heat equation (c={c}) approaching steady solution")
plt.legend()
plt.tight_layout()
plt.show()

print("dt used:", dt)
for ts in snap_times:
    key = round(ts, 4)
    if key in snapshots:
        err = np.max(np.abs(snapshots[key] - u_steady))
        print(f"t={ts:.1f}s: max|u - u_steady| ≈ {err:.3f}")
