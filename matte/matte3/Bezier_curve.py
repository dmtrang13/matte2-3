import sympy as sp
from scipy.integrate import quad

t = sp.symbols("t")

a = 1.00005507808
b = 0.55342925736
c = 0.99873327689

x = 1-(c*(1-t)**3 + 3*b*t*(1-t)**2)
y = 1-(a*t**3 + 3*b*t**2*(1-t))

dxdt = sp.diff(x, t)
dydt = sp.diff(y, t)
speed = sp.sqrt(dxdt**2 + dydt**2)
ds = sp.lambdify(t, speed, "numpy")

val, err = quad(ds, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12)
print(val, err)

sp.plot_parametric((x,y), (t,0,1))