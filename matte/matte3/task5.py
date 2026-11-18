import numpy as np
from scipy.spatial import Delaunay

# Sett opp antall tilfeldige samplepunkter vi bruker i integralet.
N = 100

# Generer N punkter i enheten [0, 1]^3 og skaler hver akse til riktig intervall.
pts = np.random.rand(N, 3)
pts[:, 0] *= 2*np.pi   # theta-verdier i [0, 2π]
pts[:, 1] *= 2.0       # r-verdier i [0, 2]
pts[:, 2] *= 4.0       # z-verdier i [0, 4]

# Legg til alle hjørnene av integrasjonsboksen slik at trianguleringen dekker hele volumet.
corners = np.array([
    [0.0,       0.0, 0.0],
    [0.0,       0.0, 4.0],
    [0.0,       2.0, 0.0],
    [0.0,       2.0, 4.0],
    [2*np.pi,   0.0, 0.0],
    [2*np.pi,   0.0, 4.0],
    [2*np.pi,   2.0, 0.0],
    [2*np.pi,   2.0, 4.0],
])
pts = np.vstack([pts, corners])

# Bygg en 3D Delaunay-triangulering; den deler volumet inn i tetraedre basert på punktene.
tri = Delaunay(pts)

# Plukk ut koordinatene til hvert hjørne i alle tetraedrene.
a = pts[tri.simplices[:, 0]]
b = pts[tri.simplices[:, 1]]
c = pts[tri.simplices[:, 2]]
d = pts[tri.simplices[:, 3]]
n = len(a)

# Finn centroid (midtpunkt) til hvert tetraeder og ekstraher r-koordinaten i centroiden.
avg = (a + b + c + d) / 4.0
rbar = avg[:, 1]  # andre kolonne representerer r

# Beregn kantvektorer, sett dem i en matrix, og finn volumet til hvert tetraeder.
u = b - a
v = c - a
w = d - a
M = np.stack([u, v, w], axis=2)          # form (n, 3, 3)
dV = np.abs(np.linalg.det(M)) / 6.0      # tetraedervolum

# Estimer integralet ved å summere rbar * volum for alle tetraedre.
svar = np.sum(dV * rbar)
print(svar)
