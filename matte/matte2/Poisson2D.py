import numpy as np
import matplotlib.pyplot as plt

def poisson_solver(nx=50, ny=20, tol=1e-5, max_iter=5000):
    # Grensebetingelser
    x_min, x_max = -5, 5
    y_min, y_max = 0, 2
    
    # Grid avstand
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    
    # Lager grid
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    
    # Initialiserer array
    u = np.zeros((nx, ny))
    
    # Tilpasser for randbetingelser
    u[0, :] = np.sin(2 * np.pi * y)  # Venstre 
    u[-1, :] = np.sin(2 * np.pi * y) # Høyre 
    u[:, 0] = 0                      # Ned 
    u[:, -1] = np.sin(np.pi * x)     # Opp 
    
    # Iterativ Gauss-Seidel metoden
    for iteration in range(max_iter):
        u_old = u.copy()
        
        # Oppdaterer indre noder
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i, j] = (dy**2 * (u[i+1, j] + u[i-1, j]) + dx**2 * (u[i, j+1] + u[i, j-1])) / (2 * (dx**2 + dy**2))
        
        # Sjekke for konvergering
        diff = np.max(np.abs(u - u_old))
        if diff < tol:
            print(f'Konvergerer etter {iteration} iterasjoner.')
            break
    
    return x, y, u

# Løser
x, y, u = poisson_solver()

# Plott
X, Y = np.meshgrid(x, y, indexing='ij')
plt.figure(figsize=(8,4))
plt.contourf(X, Y, u, levels=50, cmap='jet')
plt.colorbar(label='u(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerisk løsning av Poissonligning i 2D')
plt.show()