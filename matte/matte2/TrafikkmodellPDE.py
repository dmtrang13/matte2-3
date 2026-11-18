import numpy as np
import matplotlib.pyplot as plt


u_max = 0.202  # Maximum tetthet (biler/meter) basert på en typisk billengde
v_max = 16.67  # Maximum hastighet (m/s) basert på en grense på 60 km/h
N = 200  # Antall punkter i rommet
M = 1000  # Antall steg i tid
x_min, x_max = -1000, 1000  # Rommet
T = 100  # Total tid (sekunder)
dx = (x_max - x_min) / (N - 1)  # Steg i rommet
dt = T / M  # Steg i tid
x = np.linspace(x_min, x_max, N)  # Grid til rommet
t = np.linspace(0, T, M)  # Grid til tiden

# Hastighetsfunksjon
def v(u, p):
    return v_max * (1 - (u / u_max) ** p)

# Trafikkfluks-funksjon
def J(u, p):
    return u * v(u, p)

# Lax-Friedrichs metode
def lax_friedrichs(u0, p, N, M, dx, dt):
    u = np.zeros((M, N))
    u[0, :] = u0(x)  # Initialbetingelse

    for n in range(M - 1):
        for j in range(1, N - 1):
            u[n + 1, j] = 0.5 * (u[n, j + 1] + u[n, j - 1]) - (dt / (2 * dx)) * (
                J(u[n, j + 1], p) - J(u[n, j - 1], p)
            )

        # Randebetingelser
        u[n + 1, 0] = u_max  # Venstre 
        u[n + 1, -1] = 0  # Høyre

    return u

# Initialbetingelse med en mykere transisjon
def u0_smooth(x):
    return u_max * (1 - np.exp(x / 300)) * (x < 0)  # Mykere transisjon ved x=0

# Løser for bevegelsen til bilen ved x = -990m (istedenfor -1000)
def single_car_trajectory(U, x, dt, p):
    """
    Computes the position P(t) and velocity V(t) of a single car starting at x=-990.
    - U: Precomputed traffic density U[n, j] for all times
    - x: Spatial grid
    - dt: Time step
    - p: Parameter p in velocity function

    Returns P (position over time) and V (velocity over time).
    """
    Nt, Nx = U.shape
    P = np.zeros(Nt)
    V = np.zeros(Nt)

    # Justerer initialbetingelse til -990 isteden for -1000 (grafene ligger utenfor -1000)
    P[0] = -990
    V[0] = 0

    for n in range(1, Nt):
        # Finn det nærmeste gridpunktet til P[n-1]
        idx = np.argmin(np.abs(x - P[n-1]))
        u_local = U[n-1, idx]  # Get the density at the car's position

        # Oppdaterer hastighet og posisjon
        V[n] = v_max * (1 - (u_local / u_max) ** p)
        P[n] = P[n-1] + dt * V[n]

    return P, V

# Løs for ulike p-verdier
p_values = [1, 2, 5]
solutions_smooth = {p: lax_friedrichs(u0_smooth, p, N, M, dx, dt) for p in p_values}
trajectories_smooth = {p: single_car_trajectory(solutions_smooth[p], x, dt, p) for p in p_values}

# Plot posisjon og hastighet for hver p med oppdatert startposisjon
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

for p in p_values:
    P, V = trajectories_smooth[p]
    time_axis = np.arange(len(P)) * dt

    # Plot posisjon
    axs[0].plot(time_axis, P, label=f"p = {p}")

    # Plot hastighet
    axs[1].plot(time_axis, V, label=f"p = {p}")

axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Position P(t) (m)")
axs[0].set_title("Position of a single car over time")
axs[0].legend()
axs[0].grid()

axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Velocity V(t) (m/s)")
axs[1].set_title("Velocity of a single car over time")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
