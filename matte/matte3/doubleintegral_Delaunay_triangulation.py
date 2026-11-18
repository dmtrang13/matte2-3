import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay

pts = np.random.rand(150,2)
pts[:,0] *= 3.0
pts[0] = [0,0]
pts[1] = [3,0]
pts[2] = [0,1]
pts[3] = [3,1]

pts[4:20,0] = 3.0
pts[21:40,0] = 0.0
pts[41:60,1] = 1.0
pts[61:80,1] = 0.0
tri = Delaunay(pts)

plt.triplot(pts[:,0], pts[:,1], tri.simplices)
plt.show()

p = pts[tri.simplices[:,0]]
q = pts[tri.simplices[:,1]]
r = pts[tri.simplices[:,2]]
n = len(p)

avg = 1/3 * (p + q + r)
x = avg[:,0]
y = avg[:,1]
f = x*x*y

u = q - p
v = r - p

Muv = np.column_stack((u,v)).reshape(n,2,2)
dA = np.abs(1/2*np.linalg.det(Muv))

svar = np.sum(dA*f)
print(svar)