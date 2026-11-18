import numpy as np
import matplotlib.pyplot as plt

# vi lager verdiene for x og y
x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)
# vi lager rutenettet med 100x100 = 10.000 punkter
X, Y = np.meshgrid(x,y)
# regner sin(pi*x*y) på disse 10.000 punktene
Z = np.sin(np.pi * X * Y)
# vi tar differansene langs x-aksen, dZx[i,j] = Z[i,j+1] - Z[i,j]
# j-koordinat tilsvarer x, se kommentaren "Advarsel: Arrayer, rutenett og meshgrid"
dZx = Z[:,1:] - Z[:,:-1] # dZx = np.diff(Z, axis=1) kunne også blitt brukt
# tar differansen dX[i,j] = X[i,j+1]-X[i,j] langs y-aksen
dX = X[:,1:] - X[:,:-1]
print(dX)

# vi regner den deriverte i x-retning
# vi kan erstatte dX med et tall h hvis avstanden mellom punktene i x er konstant lik h, slik de er her
Zx = dZx / dX
# Tilsvarende beregning i y-retning
dZy = Z[1:,:] - Z[:-1,:]
dY = Y[1:,:] - Y[:-1,:] # Y = np.diff(Y, axis=0) kunne også blitt brukt
Zy = dZy / dY
# Vi plotter f_x til venstre og f_y til høyre
# vi må bruke X[:,0:-1] og tilsvarende på Y siden vi mistet en x-verdi da vi brukte np.diff
fig,(ax1, ax2) = plt.subplots(1,2,subplot_kw ={"projection":"3d"}, figsize=(15,10))
ax1.plot_surface(X[:,:-1], Y[:,:-1], Zx)
ax2.plot_surface(X[:-1,:], Y[:-1,:], Zy)
plt.show()