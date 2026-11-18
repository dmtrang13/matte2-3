import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# vi lager verdiene av x og t som vi skal bruke
x = np.arange(0, 2*np.pi, 0.01)
t = np.arange(0, 5, 0.05)

# vi lager rutenettet
X, T = np.meshgrid(x,t)
# vi regner ut funksjonen på rutenettet
u = np.cos(X+T)
# vi plotter funksjonen

fig, ax = plt.subplots()
#
# Vi lager en animasjon
#
# initialplott, for tid y=0
line, = ax.plot(x, u[0,:])
def animate(i):
 line.set_ydata(u[i,:])
 return line,
# interval bestemmer hvor fort det vises
# save_count hvor mange tider t_i som inngår
ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, save_count=t.size)
HTML(ani.to_jshtml())