import numpy as np
import matplotlib.pyplot as plt

Nx = Ny = 121                      # resolution
x = np.linspace(0.0, 40.0, Nx)     # x in [0, 40]
y = np.linspace(0.0, 40.0, Ny)     # y in [0, 40]  (0=south, 40=north)

u = np.zeros((Nx, Ny))

# Dirichlet
u[0, :] = 20.0 * (y / 40.0)

u[-1, :] = 0.0

u[:, 0] = 0.0

north = np.zeros(Nx)
mask = (x <= 20.0)
north[mask] = 20.0 * (1.0 - x[mask] / 20.0)
north[~mask] = 0.0
u[:, -1] = north

# Jacobi iteration
u_old = u.copy()
tol = 1e-6
max_iter = 20000

for it in range(max_iter):
    u_new = u_old.copy()
    u_new[1:-1, 1:-1] = 0.25 * (
        u_old[2:, 1:-1] + u_old[:-2, 1:-1] + u_old[1:-1, 2:] + u_old[1:-1, :-2]
    )
    # re-apply boundaries
    u_new[0, :]  = 20.0 * (y / 40.0)  # west
    u_new[-1, :] = 0.0                # east
    u_new[:, 0]  = 0.0                # south
    u_new[:, -1] = north              # north

    err = np.max(np.abs(u_new - u_old))
    u_old = u_new
    if err < tol:
        print(f"Converged in {it+1} iterations, max change {err:.2e}")
        break

u = u_old

plt.figure(figsize=(6, 5))
plt.imshow(u.T, origin='lower', extent=[0, 40, 40, 0], aspect='equal', cmap='inferno')
plt.colorbar(label='Temperature (°C)')
plt.title('Steady-state temperature (Laplace)')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
