# Beregning av arealet av en disk med radius 1.0
import random as rnd
import numpy as np
rnd.seed(0)

numSim = 1000000
numHit = 0
xmin = -2.0
xmax = 4.0
ymin = -2.0
ymax = 2.0
sum_f = 0

'''
for i in range(numSim):
    x = rnd.random() * (xmax - xmin) + xmin 
    y = rnd.random() * (ymax - ymin) + ymin 
    d = x*x + y*y
    if(d < 4.0):
        numHit += 1

area1 = (xmax - xmin) * (ymax - ymin)
print("Svar:", area1 * numHit / numSim)
'''
#   a)
for i in range(numSim):
    x = rnd.random() * (xmax - xmin) + xmin 
    y = rnd.random() * (ymax - ymin) + ymin 
    d1 = x*x + y*y 
    d2 = (x-2.0)*(x-2.0) + y*y  
    if(d1 < 4.0 or d2 < 4.0):
        numHit += 1

area = (xmax - xmin) * (ymax - ymin)
print("Svar:", area * numHit / numSim)

'''
#   b)
for i in range(numSim):
    x = rnd.random() * (xmax - xmin) + xmin # -1.0..1.0
    y = rnd.random() * (ymax - ymin) + ymin # -1.0..1.0
    d = x*x + y*y
    if(d <= 4.0):
        f = x*x + y*y + 2
        sum_f += f
volum = (xmax - xmin) * (ymax - ymin) * sum_f/numSim
print("Svar:", volum * numHit / numSim)
'''