import matplotlib.pyplot as plt # Pakke brukt til plotting
import numpy as np # Pakke brukt til numeriske beregninger

# a) Lag 102 punkter i intervallet som starter 0 og 2pi.
t = np.linspace(0,2*np.pi, 102)

# b) Lag x- og y-koordinaten til kurven gamma.
gamma_1 = np.exp(t)
gamma_2 = np.cos(2*t)

plt.plot(gamma_1, gamma_2)
plt.show()