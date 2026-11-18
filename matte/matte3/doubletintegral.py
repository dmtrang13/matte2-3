import numpy as np;

xmin = -2.0
xmax = 2.0
ymin = -1.0
ymax = 1.0

numx = 1000
numy = 1000
dx = (xmax - xmin) / numx
dy = (ymax - ymin) / numy
dA = dx*dy

xv = np.linspace(xmin, xmax, numx, endpoint=False)
yv = np.linspace(ymin, ymax, numy, endpoint=False)
x, y = np.meshgrid(xv, yv)

svar = np.sum(np.sin(x**2+y))*dA

print(svar)