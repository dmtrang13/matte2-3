# (a)-(e) Vector field plots, trajectories, potential, and divergence
# NOTE: This notebook generates figures for (a), (b), and (d), and prints symbolic answers for (c) and (e).
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------
# Define the vector field v(x,y; L, M)
# v = L * r1/|r1|^2 - M * r2/|r2|^2
# with r1=(x-1, y), r2=(x+1, y)
# ----------------------------
def vfield(x, y, L=1.0, M=3.0):
    # Avoid the singularities by adding a tiny epsilon in denominators if needed
    r1x, r1y = x - 1.0, y
    r2x, r2y = x + 1.0, y
    d1 = r1x*r1x + r1y*r1y
    d2 = r2x*r2x + r2y*r2y
    # Vector field components
    vx = L * r1x / d1 - M * r2x / d2
    vy = L * r1y / d1 - M * r2y / d2
    return vx, vy

# ----------------------------
# (a) Streamplot and quiver for L=1, M=3
# ----------------------------
L, M = 1.0, 3.0
xmin, xmax = -4.0, 4.0
ymin, ymax = -3.0, 3.0
nx, ny = 200, 150

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

VX, VY = vfield(X, Y, L, M)

# Mask out a small disk around the singularities to improve plotting
mask = ( (X-1.0)**2 + Y**2 < 0.03**2 ) | ( (X+1.0)**2 + Y**2 < 0.03**2 )
VX_masked = np.ma.masked_where(mask, VX)
VY_masked = np.ma.masked_where(mask, VY)

# Streamplot
plt.figure(figsize=(7, 4.5))
plt.streamplot(X, Y, VX_masked, VY_masked, density=1.2)
plt.scatter([1, -1], [0, 0], s=60, marker='o')  # source at (1,0), sink at (-1,0)
plt.title("Streamplot of v(x,y) for L=1, M=3")
plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal'); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
plt.show()

# Quiver
step = 6  # thin the grid for quiver
plt.figure(figsize=(7, 4.5))
plt.quiver(X[::step, ::step], Y[::step, ::step], VX_masked[::step, ::step], VY_masked[::step, ::step])
plt.scatter([1, -1], [0, 0], s=60, marker='o')
plt.title("Quiver of v(x,y) for L=1, M=3")
plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal'); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
plt.show()

# ----------------------------
# (b) Trajectories from the source using SciPy's solve_ivp and Forward Euler
# ----------------------------
def rhs(t, Y):
    x, y = Y
    vx, vy = vfield(x, y, L, M)
    return [vx, vy]

# Initial points near the source at (1,0); offset slightly to avoid singularity
inits = [
    (1.05, 0.00),
    (1.05, 0.20),
    (1.05, -0.20),
    (1.10, 0.40),
    (1.10, -0.40)
]

t_span = (0.0, 20.0)
t_eval = np.linspace(t_span[0], t_span[1], 2000)

plt.figure(figsize=(7, 4.5))
for x0, y0 in inits:
    sol = solve_ivp(rhs, t_span, [x0, y0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
    plt.plot(sol.y[0], sol.y[1])
plt.scatter([1, -1], [0, 0], s=60, marker='o')
plt.title("Trajectories (solve_ivp) for particles starting near (1,0); L=1, M=3")
plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal'); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
plt.show()

# Forward Euler implementation
def euler_forward(x0, y0, h=0.01, nsteps=5000):
    xs = np.empty(nsteps+1); ys = np.empty(nsteps+1)
    xs[0], ys[0] = x0, y0
    for k in range(nsteps):
        vx, vy = vfield(xs[k], ys[k], L, M)
        xs[k+1] = xs[k] + h*vx
        ys[k+1] = ys[k] + h*vy
        # Stop if we get too close to singularities to avoid blow-up
        if ( (xs[k+1]-1.0)**2 + ys[k+1]**2 < 1e-4 ) or ( (xs[k+1]+1.0)**2 + ys[k+1]**2 < 1e-4 ):
            xs = xs[:k+2]; ys = ys[:k+2]
            break
    return xs, ys

plt.figure(figsize=(7, 4.5))
for x0, y0 in inits:
    xs, ys = euler_forward(x0, y0, h=0.01, nsteps=20000)
    plt.plot(xs, ys, linewidth=1.0)
plt.scatter([1, -1], [0, 0], s=60, marker='o')
plt.title("Trajectories (Forward Euler) for particles near (1,0); L=1, M=3")
plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal'); plt.xlim(xmin, xmax); plt.ylim(ymin, ymax)
plt.show()

# ----------------------------
# (c) Potential phi(x,y) for general L, M (symbolic expression as a string)
# ----------------------------
phi_symbolic = "(L/2)*log((x-1)**2 + y**2) - (M/2)*log((x+1)**2 + y**2)"
print("Symbolic potential (no constant):")
print("phi(x,y) =", phi_symbolic)

# ----------------------------
# (d) Contours of potential for L=1, M=1 using matplotlib
# ----------------------------
L1, M1 = 1.0, 1.0
def phi(x, y, L=L1, M=M1):
    return 0.5*np.log((x-1.0)**2 + y**2) - 0.5*np.log((x+1.0)**2 + y**2)

x2 = np.linspace(-3.0, 3.0, 400)
y2 = np.linspace(-3.0, 3.0, 400)
X2, Y2 = np.meshgrid(x2, y2)
PHI = phi(X2, Y2, L1, M1)

plt.figure(figsize=(7, 5))
cs = plt.contour(X2, Y2, PHI, levels=15)
plt.clabel(cs, inline=True, fontsize=8)
plt.scatter([1, -1], [0, 0], s=60, marker='o')
plt.title("Level curves of phi for L=1, M=1")
plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal'); plt.xlim(-3, 3); plt.ylim(-3, 3)
plt.show()

# ----------------------------
# (e) Divergence at general (x,y) != (±1,0)
# Compute analytically using partial derivatives to confirm it's zero
import sympy as sp
x_s, y_s, L_s, M_s = sp.symbols('x y L M', real=True)
vx_expr = L_s*(x_s-1)/((x_s-1)**2 + y_s**2) - M_s*(x_s+1)/((x_s+1)**2 + y_s**2)
vy_expr = L_s*y_s/((x_s-1)**2 + y_s**2) - M_s*y_s/((x_s+1)**2 + y_s**2)
div_expr = sp.simplify(sp.diff(vx_expr, x_s) + sp.diff(vy_expr, y_s))
print("Symbolic divergence (away from the singular points):")
print(div_expr.simplify())
