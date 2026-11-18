import numpy as np
import matplotlib.pyplot as plt

Nx = Ny = 201
x = np.linspace(0.0, 1.0, Nx)
y = np.linspace(0.0, 1.0, Ny)

u = np.zeros((Nx, Ny), dtype=float)

u[0, :] = 20.0 * y          # vest: 0 -> 20
u[-1, :] = 0.0              # øst: 0
u[:, 0] = 0.0               # sør: 0
north = np.zeros(Nx)        # nord: 20 -> 0 på [0, 0.5], ellers 0
mask = x <= 0.5
north[mask] = 20.0 * (1.0 - 2.0 * x[mask])
u[:, -1] = north

omega = 1.6         # 1.2–1.8 er ofte fint; senk hvis du ser NaN/Inf
tol = 1e-6
max_iter = 50000
report_every = 200

ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
interior = (ii > 0) & (ii < Nx-1) & (jj > 0) & (jj < Ny-1)
red_mask   = interior & ((ii + jj) % 2 == 0)
black_mask = interior & ((ii + jj) % 2 == 1)

def apply_bc(U):
    U[0, :]  = 20.0 * y
    U[-1, :] = 0.0
    U[:, 0]  = 0.0
    U[:, -1] = north

def residual_inf_norm(U):
    R = (U[2:, 1:-1] + U[:-2, 1:-1] + U[1:-1, 2:] + U[1:-1, :-2] - 4.0 * U[1:-1, 1:-1])
    return np.max(np.abs(R))

apply_bc(u)

for it in range(1, max_iter + 1):
    avg = np.zeros_like(u)
    avg[1:-1, 1:-1] = 0.25 * (
        u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
    )
    u[red_mask] = (1 - omega) * u[red_mask] + omega * avg[red_mask]
    apply_bc(u)

    avg[1:-1, 1:-1] = 0.25 * (
        u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
    )
    u[black_mask] = (1 - omega) * u[black_mask] + omega * avg[black_mask]
    apply_bc(u)

    if not np.isfinite(u).all():
        raise RuntimeError("Divergens: NaN/Inf i løsningen. Senk omega, f.eks. 1.2–1.6.")
    res = residual_inf_norm(u)
    if it % report_every == 0 or res < tol:
        print(f"iter {it}: ||res||_inf = {res:.3e}")
    if res < tol:
        print(f"Konvergerte etter {it} iterasjoner.")
        break

plt.figure(figsize=(6,5))
im = plt.imshow(u.T, origin='lower', extent=[0,1,0,1], aspect='equal', cmap='inferno')
plt.colorbar(im, label='Temperatur')
plt.title('2D Laplace – FVM (Red–Black SOR)')
plt.xlabel('x (vest→øst)')
plt.ylabel('y (sør→nord)')
plt.tight_layout()
plt.show()