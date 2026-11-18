import numpy as np
import matplotlib.pyplot as plt

# lager 5 punkter mellom -1 og 0 for x og mellom 0 og 1 for y
x = np.linspace(-1,0,5)
y = np.linspace(0,1,5)
# lager et rutenett med 5x5 = 25 punkter
X1, Y1 = np.meshgrid(x,y)
# lager samme rutenett, med motsett rekkefølge
X2, Y2 = np.meshgrid(x,y,indexing='ij')
# print ut X-verdiene i den vanlige - de følger den vanlige x-aksen
print(X1)
# hvis du printer ut X-verdiene for den andre muligheten, vil du kansje ikke få hva du forventet
print(X2)
# vi lager verdiene for x og y med 100 verdier for hver
x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)
# lager rutenettet med 100x100 = 10.000 punkter
X, Y = np.meshgrid(x,y)
# finn sin(pi*x*y) på disse 10.000 punktene
Z = np.sin(np.pi * X * Y)
# plotter de 10.000 funksjonsverdiene
fig,ax2 = plt.subplots(subplot_kw ={"projection":"3d"}, figsize=(10,8))
ax2.plot_surface(X, Y, Z)