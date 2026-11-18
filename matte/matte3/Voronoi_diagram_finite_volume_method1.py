
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay, Voronoi
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from shapely.geometry import Polygon, box
from matplotlib.collections import PolyCollection

# Punkter i området
rng = np.random.default_rng(7)

n_inner = 130
n_side = 15

t = np.linspace(0, 1, n_side)

bottom = np.column_stack((t, np.zeros_like(t)))
top = np.column_stack((t, np.ones_like(t)))
left = np.column_stack((np.zeros_like(t), t))
right = np.column_stack((np.ones_like(t), t))

# Fjern doble hjørnepunkter
boundary_points = np.unique(
    np.vstack((bottom, top, left, right)),
    axis=0
)

inner_points = rng.uniform(
    0.03, 0.97, size=(n_inner, 2)
)

points = np.vstack((boundary_points, inner_points))

N = len(points)

# Finn hvilke punkter som ligger på randen
is_boundary = np.any(
    np.isclose(points, 0.0) |
    np.isclose(points, 1.0),
    axis=1
)

# Delaunay-triangulering
tri = Delaunay(points)

# Voronoi-diagram med speilpunkter
# Speilpunktene brukes til å gjøre Voronoi-cellene
# til de opprinnelige punktene begrensede.
# Til slutt klipper vi cellene til [0,1] x [0,1].
eps = 1e-4

ghosts = np.vstack((
    np.column_stack((-points[:, 0] - eps,
                      points[:, 1])),

    np.column_stack((2 - points[:, 0] + eps,
                      points[:, 1])),

    np.column_stack((points[:, 0],
                     -points[:, 1] - eps)),

    np.column_stack((points[:, 0],
                      2 - points[:, 1] + eps))
))

vor = Voronoi(np.vstack((points, ghosts)))

domain = box(0, 0, 1, 1)

cells = []

for i in range(N):
    region_index = vor.point_region[i]
    vertex_indices = vor.regions[region_index]

    if -1 in vertex_indices or len(vertex_indices) == 0:
        raise ValueError("Fant en ubegrenset Voronoi-celle")

    polygon = Polygon(vor.vertices[vertex_indices])
    cells.append(polygon.intersection(domain))

# Finn Delaunay-naboer
edges = set()

for a, b, c in tri.simplices:
    for i, j in ((a, b), (b, c), (c, a)):
        edges.add(tuple(sorted((int(i), int(j)))))

# Bygg FVM-matrisen
A = lil_matrix((N, N))
b = np.zeros(N)

for i, j in edges:

    # Lengden av felles Voronoi-kant
    common = cells[i].boundary.intersection(
        cells[j].boundary
    )
    L_ij = common.length

    if L_ij < 1e-10:
        continue

    # Avstanden mellom beregningspunktene
    d_ij = np.linalg.norm(points[j] - points[i])

    # FVM-koeffisient
    w = L_ij / d_ij

    # Fluksbidrag til begge kontrollvolumene
    A[i, i] += w
    A[j, j] += w
    A[i, j] -= w
    A[j, i] -= w

# Dirichlet-randbetingelser
def u_exact(x, y):
    return x**2 - y**2

for i in np.flatnonzero(is_boundary):
    A.rows[i] = [int(i)]
    A.data[i] = [1.0]
    b[i] = u_exact(*points[i])

# Løs systemet
u = spsolve(A.tocsr(), b)

u_true = u_exact(points[:, 0], points[:, 1])

error = np.abs(u - u_true)

print("Antall punkter:", N)
print("Antall Delaunay-kanter:", len(edges))
print("Maksimal absolutt feil:", np.max(error))
print("Gjennomsnittlig absolutt feil:", np.mean(error))

# Plot Delaunay-trianguleringen
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].triplot(
    points[:, 0],
    points[:, 1],
    tri.simplices,
    color="gray",
    linewidth=0.6
)

axs[0, 0].scatter(
    points[:, 0],
    points[:, 1],
    s=8,
    color="red"
)

axs[0, 0].set_title("Delaunay-triangulering")

# Plot Voronoi-kontrollvolumene
polygons = [
    np.asarray(cell.exterior.coords)
    for cell in cells
]

collection = PolyCollection(
    polygons,
    facecolors="lightblue",
    edgecolors="black",
    linewidths=0.5
)

axs[0, 1].add_collection(collection)

axs[0, 1].scatter(
    points[:, 0],
    points[:, 1],
    s=7,
    color="red"
)

axs[0, 1].set_title("Voronoi-kontrollvolumer")

# Plot numerisk løsning
plot1 = axs[1, 0].tricontourf(
    points[:, 0],
    points[:, 1],
    tri.simplices,
    u,
    levels=30,
    cmap="viridis"
)

fig.colorbar(plot1, ax=axs[1, 0])
axs[1, 0].set_title("Numerisk FVM-løsning")

# Plot absolutt feil
plot2 = axs[1, 1].tricontourf(
    points[:, 0],
    points[:, 1],
    tri.simplices,
    error,
    levels=30,
    cmap="magma"
)

fig.colorbar(plot2, ax=axs[1, 1])
axs[1, 1].set_title("Absolutt feil")

for ax in axs.flat:
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.tight_layout()
plt.show()
