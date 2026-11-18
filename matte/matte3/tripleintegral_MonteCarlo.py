import numpy as np

rng = np.random.default_rng(0)   # for reproduserbarhet
N = 300_000                      # øk/juster fritt

xyz = rng.random((N, 3))         # uniformt i [0,1]^3
x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]

inside = (y <= x**2) & (z <= y**2)
estimate = inside.mean()         # volum ~= andel treff

print(estimate)         