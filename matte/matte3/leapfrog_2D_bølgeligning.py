import numpy as np
import matplotlib.pyplot as plt

#parametere
c = 1.0
N = 50

x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)

dx = x[1] - x[0]
dy = y[1] - y[0]

#stabil tidssteglengde
dt = 0.9 * dx / (c * np.sqrt(2))

X, Y = np.meshgrid(x, y)

#initialbetingelser
u0 = np.exp(-50 * (X**2 + Y**2))

# Randbetingelser: u = 0
u0[0, :] = 0
u0[-1, :] = 0
u0[:, 0] = 0
u0[:, -1] = 0

#nabla^2u=u_{xx}+u_{yy}
def laplace(u):
    result = np.zeros_like(u)

    #FDM
    result[1:-1, 1:-1] = (
        (u[2:, 1:-1] - 2*u[1:-1, 1:-1]
         + u[:-2, 1:-1]) / dx**2
        +
        (u[1:-1, 2:] - 2*u[1:-1, 1:-1]
         + u[1:-1, :-2]) / dy**2
    )

    return result

#første tidssteg (initialhastighet = 0)
u_prev = u0.copy()
u = u0 + 0.5 * c**2 * dt**2 * laplace(u0)

#tidspunkter vi ønsker å plotte
times = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
snapshots = {0.0: u0.copy()}

t = dt
t_end = max(times)

#leapfrog
while t < t_end + dt:

    for target in times[1:]:
        if target not in snapshots and t >= target:
            snapshots[target] = u.copy()

    #leapfrog 
    #u^{n+1} = 2u^n - u^{n-1} + c^2\Delta t^2 \nabla_h^2 u^n
    u_next = 2*u - u_prev + c**2 * dt**2 * laplace(u)

    #randbetingelser
    u_next[0, :] = 0
    u_next[-1, :] = 0
    u_next[:, 0] = 0
    u_next[:, -1] = 0

    u_prev = u
    u = u_next
    t += dt

#plot
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

vmax = max(np.max(np.abs(s)) for s in snapshots.values())

#løsninger ved bestemte tidspunkter
for ax, target in zip(axes.flat, times):
    im = ax.imshow(
        snapshots[target],
        extent=[-1, 1, -1, 1],
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax
    )

    ax.set_title(f"t = {target}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.colorbar(im, ax=axes.ravel().tolist(),
             label="u(x,y,t)", shrink=0.8)

plt.show()

print("dx =", dx)
print("dt =", dt)
