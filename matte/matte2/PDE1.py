import numpy as np
import matplotlib.pyplot as plt

# vi lager x og y-verdiene vi skal bruke
x = np.linspace(-1, 1, 256)
y = np.linspace(-1, 1, 256)
# vi lager rutenettet
X, Y = np.meshgrid(x, y)
# vi regner ut funksjonen på alle punktene i rutenettet
Z = np.cos(np.pi*(X+Y))
# vi plotter grafen til funksjonen
fig, ax2 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15, 15))
ax2.plot_surface(X, Y, Z)
plt.show()