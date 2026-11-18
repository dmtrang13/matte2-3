import numpy as np
import matplotlib.pyplot as plt

# lag 100 punkter mellom 0 og 1 langs x-aksen
x = np.linspace(0,1,100)
# vi regner f(x) på alle punktene
f = np.sin(np.pi * x)
# vi lager df hvor df[i] = f[i+1]-f[i]
# funksjonen np.diff() kan også brukes
df = f[1:] - f[:-1]
# vi lager dx hvor dx[i] = x[i+1] - x[i]
# (akkurat her så kommer alle elementene til å være h)
dx = x[1:] - x[:-1]
# vi lager den deriverte, dvs. g[i] = df[i]/dx[i]
g = df / dx
# vi har kun 99 punkter siden np.diff(f) må stoppe utregningen for df[i] = f[i+1] - f[i] ved i=98.
plt.plot(x[:-1],g)