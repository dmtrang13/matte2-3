import sympy as sp
x,y = sp.symbols("x,y")

f = x*x * y

svar = sp.integrate(f, x, y)

print(svar)