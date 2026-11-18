import random as rnd
numSim = 10000
numHit = 0
xmin = -1.0
xmax = 1.0
ymin = -1.0
ymax = 1.0

for i in range(numSim):
    x = rnd.random() * (xmax - xmin) + xmin # -1.0..1.0
    y = rnd.random() * (ymax - ymin) + ymin # -1.0..1.0
    d = x*x + y*y
    if(d < 1.0):
        numHit += 1

area = (xmax - xmin) * (ymax - ymin)
print("Svar:", area * numHit / numSim)