import sympy as sp

x, y = sp.symbols("x y", real=True)

# Eksakt strømfunksjon
psi = -sp.sin(sp.pi*x)*sp.sin(sp.pi*y)/sp.pi**2

# Sjekk Poissonligningen
laplace_psi = sp.diff(psi, x, 2) + sp.diff(psi, y, 2)

print("Laplace(psi) =", sp.simplify(laplace_psi))

# Finn hastighetskomponentene
u = sp.diff(psi, y)
v = -sp.diff(psi, x)

print("u =", sp.simplify(u))
print("v =", sp.simplify(v))

# Kontroller divergens
div_F = sp.diff(u, x) + sp.diff(v, y)

print("Divergens =", sp.simplify(div_F))

# Kontroller vortisitet
omega = sp.diff(v, x) - sp.diff(u, y)

print("Vortisitet =", sp.simplify(omega))