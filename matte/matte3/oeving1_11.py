import numpy as np
from scipy import integrate

a = 10
b = -10
numval = 1000
x = np.linspace(b,a,numval, endpoint=False)
dx = (a - b) / numval
y = np.exp(1-x**2)
svar = np.sum(y) * dx
print(svar)

f = lambda x : np.exp(1-x**2)

svar, feil = integrate.quad(f, -10.0, 10.0)