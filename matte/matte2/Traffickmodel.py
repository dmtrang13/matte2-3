import numpy as np
import matplotlib.pyplot as plt

# Parameters
u_max = 0.202  # maximum density (cars/meter)
v_max = 16.67  # maximum velocity (m/s) based on speed limit of 60 km/h
N = 200  # number of spatial points
M = 1000  # number of time steps
x_min, x_max = -1000, 1000  # spatial domain
T = 100  # total time (seconds)
dx = (x_max - x_min) / (N - 1)  # spatial step
dt = T / M  # time step

x = np.linspace(x_min, x_max, N)  # spatial grid
t = np.linspace(0, T, M)  # time grid

# Velocity function
def v(u, p):
    return v_max * (1 - (u / u_max) ** p)

# Traffic flux function
def J(u, p):
    return u * v(u, p)

# Lax-Friedrichs method
def lax_friedrichs(u0, p, N, M, dx, dt):
    u = np.zeros((M, N))
    u[0, :] = u0(x)  # initial condition

    for n in range(M - 1):
        for j in range(1, N - 1):
            u[n + 1, j] = 0.5 * (u[n, j + 1] + u[n, j - 1]) - (dt / (2 * dx)) * (J(u[n, j + 1], p) - J(u[n, j - 1], p))

        # Boundary conditions
        u[n + 1, 0] = u_max  # left boundary
        u[n + 1, -1] = 0  # right boundary

    return u

# Initial condition
def u0(x):
    return np.where(x < 0, u_max, 0)  # cars waiting at x < 0, no cars at x >= 0

# Solve for p = 1, 2, 5
p_values = [1, 2, 5]
solutions = {p: lax_friedrichs(u0, p, N, M, dx, dt) for p in p_values}

# Plot results for each p
for p in p_values:
    plt.figure(figsize=(10, 6))

    for t_step in range(0, M, M // 10):  # plot every 10th time step
        plt.plot(x, solutions[p][t_step, :], label=f"t = {t[t_step]:.1f} s")

    plt.xlabel("Position \( x \) (meters)")
    plt.ylabel("Density \( u(x, t) \) (cars/meter)")
    plt.title(f"Traffic Density Evolution (p = {p})")
    plt.legend()
    plt.grid()
    plt.show()
