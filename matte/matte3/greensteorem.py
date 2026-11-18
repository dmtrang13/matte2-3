import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# parametrisering
t = sp.symbols('t', real=True)
x = 2*sp.cos(t)**3
y = 8*sp.sin(t)**3

# Greens teorem F = (-y, 0):  Area = ∮ -y dx
dx = sp.diff(x, t)
integrand = -y*dx
area_exact = sp.simplify(sp.integrate(integrand, (t, 0, 2*sp.pi)))
area_float = float(area_exact.evalf())

print(area_exact, area_float)

# Plot r(t) = (2 cos^3 t, 8 sin^3 t)
ts = np.linspace(0, 2*np.pi, 800)
xs = 2*np.cos(ts)**3
ys = 8*np.sin(ts)**3

plt.figure(figsize=(5,5))
plt.plot(xs, ys, linewidth=2)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('x')
plt.ylabel('y')
plt.title('r(t) = [2 cos^3 t, 8 sin^3 t], 0 ≤ t ≤ 2π')
plt.show()
