import numpy as np
from scipy.spatial import Delaunay, Voronoi
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# pip install shapely
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon


# Domain and (non-uniform) sample points
domain_poly = Polygon([(0,0),(1,0),(1,1),(0,1)])

# Non-uniform cloud (random + a few fixed boundary points)
rng = np.random.default_rng(4)
nb = 80   # interior random points
Xint = rng.random((nb, 2))  # (0,1)^2

# Add boundary ring to ensure good boundary conformity
B = np.linspace(0,1,25)
bdry = np.vstack([
    np.c_[B, 0*B],
    np.c_[1+0*B, B],
    np.c_[B[::-1], 1+0*B],
    np.c_[0*B, B[::-1]],
])

X = np.vstack([Xint, bdry])

# Keep only points inside domain (with small buffer to include boundary)
mask = np.array([
    domain_poly.buffer(1e-12).contains(Point(p))
    for p in X
])
X = X[mask]

# Remove duplicate points at the corners
X = np.unique(X, axis=0)

# Delaunay and Voronoi
tri = Delaunay(X)
vor = Voronoi(X)


# Helper: clip Voronoi region for a site i to the domain and get polygon
def voronoi_cell_polygon(i):
    # Start with the entire computational domain
    poly = domain_poly

    pi = X[i]

    # A Voronoi cell is the set of points that are closer to pi than to any other point.
    for j in range(len(X)):
        if j == i:
            continue

        pj = X[j]

        # Midpoint and normal between the points
        mid = 0.5 * (pi + pj)
        n = pj - pi

        # Construct a large polygon that covers
        # the half-plane closest to pi
        t = np.array([-n[1], n[0]])
        t = t / np.linalg.norm(t)

        n_unit = n / np.linalg.norm(n)

        L = 10.0

        halfplane = Polygon([
            mid + L*t,
            mid - L*t,
            mid - L*t - L*n_unit,
            mid + L*t - L*n_unit
        ])

        # Only keep the portion closest to pi
        poly = poly.intersection(halfplane)

        if poly.is_empty:
            break

    return poly


# Build control volumes (restricted Voronoi cells). For robustness, drop degenerate cells.
cells = []
valid_nodes = []

for i in range(len(X)):
    poly = voronoi_cell_polygon(i)

    if (
        poly is not None
        and not poly.is_empty
        and poly.area > 1e-14
    ):
        cells.append(poly)
        valid_nodes.append(i)

valid_nodes = np.array(valid_nodes)

node_index_map = {
    gi: li for li, gi in enumerate(valid_nodes)
}

# Check that the Voronoi cells cover the domain
total_area = sum(cell.area for cell in cells)
print("Total Voronoi area =", total_area)
print("Domain area =", domain_poly.area)


# Boundary data (Dirichlet example)
# Example boundary condition: u = 0 on the whole boundary except left edge u=1
def is_on_left(p, tol=1e-10):
    return abs(p[0]-0.0) <= tol

def is_on_right(p, tol=1e-10):
    return abs(p[0]-1.0) <= tol

def is_on_bottom(p, tol=1e-10):
    return abs(p[1]-0.0) <= tol

def is_on_top(p, tol=1e-10):
    return abs(p[1]-1.0) <= tol


def dirichlet_value(xy):
    if is_on_left(xy):
        return 1.0
    return 0.0


# Face enumeration and transmissibilities
# We need interior faces between adjacent Voronoi cells and boundary faces of each cell.

A = lil_matrix(
    (len(valid_nodes), len(valid_nodes)),
    dtype=float
)
b = np.zeros(len(valid_nodes))


# Utility: nearest distance from a point to a segment (returns distance and projection point)
def point_segment_distance(p, a, b):
    ap = p - a
    ab = b - a

    t = np.clip(
        (ap @ ab) / (ab @ ab + 1e-30),
        0.0,
        1.0
    )

    q = a + t*ab

    return np.linalg.norm(p - q), q


# Precompute neighbor graph from Delaunay (who shares an edge)
neighbors = {i: set() for i in range(len(X))}

for tri_simplex in tri.simplices:
    for k in range(3):
        i = tri_simplex[k]
        j = tri_simplex[(k+1) % 3]

        neighbors[i].add(j)
        neighbors[j].add(i)


# Build a dictionary of Voronoi ridge -> segments
# For interior neighbors i<->j, the Voronoi ridge is the set of vertices in vor.ridge_vertices
# that correspond to the bisector between i and j. ridges with -1 are unbounded; our clipping fixed that.
ridge_map = {}

for (p, q), rv in zip(
    vor.ridge_points,
    vor.ridge_vertices
):
    if (
        p in node_index_map
        and q in node_index_map
    ):
        # region between p and q: build segments from rv if finite
        if -1 in rv or len(rv) < 2:
            # unbounded; our control cell clipping handles it locally via polygon boundaries
            ridge_map[(p, q)] = None
            ridge_map[(q, p)] = None

        else:
            seg = [vor.vertices[idx] for idx in rv]

            ridge_map[(p, q)] = seg
            ridge_map[(q, p)] = seg


# Loop all valid control volumes and assemble
for gi in valid_nodes:
    li = node_index_map[gi]
    Pi = np.array(X[gi])

    # Interior faces: for each neighbor gj, find the shared polygon edge between cells[li] and cells[lj]
    for gj in neighbors[gi]:
        if gj not in node_index_map:
            continue

        lj = node_index_map[gj]

        if lj <= li:
            continue  # assemble each interior edge once

        Pj = np.array(X[gj])
        dij = np.linalg.norm(Pj - Pi)

        # Find shared Voronoi face as polygon intersection boundary
        poly_i = cells[li]
        poly_j = cells[lj]

        # Use a small tolerance to handle floating-point errors
        tol = 1e-8

        shared = poly_i.boundary.intersection(
            poly_j.boundary.buffer(tol)
        )

        length = shared.length

        if length <= 1e-12:
            continue

        Tij = length / dij

        # Two-point flux: contributes +Tij on diag both cells, and -Tij off-diagonal
        A[li, li] += Tij
        A[lj, lj] += Tij
        A[li, lj] -= Tij
        A[lj, li] -= Tij

    # Boundary faces of cell i: intersect cell boundary with domain boundary
    # Extract cell edges on domain boundary
    cell_boundary = cells[li].boundary
    dom_boundary = domain_poly.boundary

    on_b = cell_boundary.intersection(dom_boundary)

    # For each boundary edge, add Dirichlet contribution via ghost-point two-point flux
    if not on_b.is_empty:
        segments = []

        if on_b.geom_type == "LineString":
            segments = [on_b]

        elif on_b.geom_type == "MultiLineString":
            segments = list(on_b.geoms)

        elif on_b.geom_type == "GeometryCollection":
            segments = [
                g for g in on_b.geoms
                if g.geom_type == "LineString"
            ]

        for seg in segments:
            a = np.array(seg.coords[0])
            bpt = np.array(seg.coords[-1])

            # length of boundary face
            Lf = np.linalg.norm(bpt - a)

            if Lf <= 1e-14:
                continue

            # distance from Pi to the boundary segment (normal projection)
            dist, proj = point_segment_distance(
                Pi, a, bpt
            )

            # Avoid division by zero for nodes on the boundary
            if dist < 1e-10:
                continue

            Tib = Lf / dist

            # Dirichlet value at the midpoint of the boundary edge
            g = dirichlet_value(0.5*(a+bpt))

            A[li, li] += Tib
            b[li] += Tib * g


# Set Dirichlet values directly for boundary nodes
for gi in valid_nodes:
    li = node_index_map[gi]
    x, y = X[gi]

    if (
        is_on_left(X[gi])
        or is_on_right(X[gi])
        or is_on_bottom(X[gi])
        or is_on_top(X[gi])
    ):
        # Left boundary = 1, other boundaries = 0
        # Corners are set to 0
        g = 1.0 if (
            is_on_left(X[gi])
            and not is_on_bottom(X[gi])
            and not is_on_top(X[gi])
        ) else 0.0

        A.rows[li] = [li]
        A.data[li] = [1.0]
        b[li] = g


# Solve
A = csr_matrix(A)
u = spsolve(A, b)

from numpy.linalg import norm

print(
    "Residual ||A*u - b|| =",
    norm(A.dot(u) - b, np.inf)
)


# u is defined on valid_nodes; build a full array if desired
U = np.full(len(X), np.nan)
U[valid_nodes] = u

# Visualise with matplotlib + shapely by coloring each cell with U[i].
print("Number of total points:", len(X))
print("Number of valid Voronoi cells:", len(valid_nodes))
print("Minimum u =", np.nanmin(U))
print("Maximum u =", np.nanmax(U))
print("Standard deviation =", np.nanstd(U))



patches = []
colors = []

for li, gi in enumerate(valid_nodes):
    xy = np.asarray(cells[li].exterior.coords)

    patches.append(MplPolygon(xy, closed=True))
    colors.append(float(U[gi]))

fig, ax = plt.subplots(figsize=(6, 6))

pc = PatchCollection(
    patches,
    cmap="inferno",
    edgecolor="black",
    linewidth=0.3
)

arr = np.asarray(colors, dtype=float)

print("Antall fargeverdier:", len(arr))
print("Antall NaN:", np.isnan(arr).sum())
print("Min:", np.nanmin(arr))
print("Max:", np.nanmax(arr))
print("Antall nullrader:", np.sum(np.diff(A.indptr) == 0))
print("Antall NaN i u:", np.isnan(u).sum())
print("Antall inf i u:", np.isinf(u).sum())

pc.set_array(arr)
pc.set_clim(0, 1)

ax.add_collection(pc)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal", "box")

plt.colorbar(pc, ax=ax, label="u")

ax.set_title(
    "Laplace on Voronoi control volumes\n"
    "(Dirichlet left=1, others=0)"
)

plt.tight_layout()
plt.show()