import sympy as sp
from scipy import integrate
import numpy as np


t = sp.symbols("t")

a = 1.00005507808
b = 0.55342925736
c = 0.99873327689

x = 1-(c*(1-t)**3 + 3*b*t*(1-t)**2)
y = 1-(a*t**3 + 3*b*t**2*(1-t))

# sp.plot_parametric((x,y), (t,0,1))

dx = sp.diff(x, t)
dy = sp.diff(y, t)
ds = sp.sqrt(dx**2 + dy**2) #curve formula

f = sp.lambdify(t, ds, "numpy") #convert to numpy function
print(integrate.quad(f, 0, 1)) #integrate the function from 0 to 1

