import numpy as np

def solve_laplace_1d(u0, u1, N=300):
    """
    Løs u'' = 0 på [0,1], u(0)=u0, u(1)=u1 med standard 3-punkts differanse.
    """
    h = 1.0 / N
    n = N - 1                      # antall indre punkter
    A = np.zeros((n, n))
    b = np.zeros(n)

    # tridiagonal: 1, -2, 1
    np.fill_diagonal(A, -2.0)
    np.fill_diagonal(A[1:], 1.0)
    np.fill_diagonal(A[:,1:], 1.0)

    # randbidrag
    b[0]  -= u0
    b[-1] -= u1

    u_int = np.linalg.solve(A, b)  # indre nodalverdier
    u = np.empty(N+1)
    u[0], u[-1] = u0, u1
    u[1:-1] = u_int
    x = np.linspace(0.0, 1.0, N+1)
    return x, u

def solve_poisson_1d(u0, u1, f, N=300):
    """
    Løs u'' + f(x) = 0 på [0,1], u(0)=u0, u(1)=u1.
    Diskret: (u_{i-1} - 2u_i + u_{i+1})/h^2 + f(x_i) = 0.
    """
    h = 1.0 / N
    n = N - 1
    x = np.linspace(0.0, 1.0, N+1)
    xi = x[1:-1]

    A = np.zeros((n, n))
    np.fill_diagonal(A, -2.0)
    np.fill_diagonal(A[1:], 1.0)
    np.fill_diagonal(A[:,1:], 1.0)

    # høyreside: ganger med h^2
    b = -(h**2) * f(xi)
    b[0]  -= u0
    b[-1] -= u1

    u_int = np.linalg.solve(A, b)
    u = np.empty(N+1)
    u[0], u[-1] = u0, u1
    u[1:-1] = u_int
    return x, u

# ---- Kjøring for oppg. 1: u''=0, u(0)=100, u(1)=200 ----
x1, u1 = solve_laplace_1d(100.0, 200.0, N=300)
i_033 = int(round(0.33 * 300))   # 0.33 = 99/300, et rutenettpunkt
print("u(0.33) for u''=0:", float(u1[i_033]))   # forventet ~133.0

# ---- Kjøring for oppg. 2: u'' + sin(x) = 0, u(0)=1.1, u(1)=1.2 ----
x2, u2 = solve_poisson_1d(1.1, 1.2, np.sin, N=300)
i_066 = int(round(0.66 * 300))   # 0.66 = 198/300, et rutenettpunkt
print("u(0.66) for u''+sin(x)=0:", float(u2[i_066]))
