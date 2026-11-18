import numpy as np
from scipy.spatial import Delaunay

#sylinder
r = 2
h = 4


n = 3000
theta = np.random.rand(n) * 2* np.pi
radius = r * np.sqrt(np.random.rand(n))
x = radius * np.cos(theta)
y = radius * np.sin(theta)
z = np.random.rand(n) * h

indre = np.column_stack((x, y, z)) #stacking the inside of the cylinder

vinkel = np.linspace(0, 2*np.pi, 100, endpoint=False) #angles for the cylinder surface to be approximated for delaunay triangulation
hoyde = np.linspace(0, h, 20) #genearets evenly spaced heights for the cylinder's height

skall = [] #empty list to store points
for z0 in hoyde: #loops thorugh all the heights of the cylinder
    for t in vinkel: #loops through all the angles of the cylinder
        skall.append([r*np.cos(t), r*np.sin(t), z0]) #add a point with the coordinate (x,y,z)
        
skall = np.array(skall) #convert the list to a numpy array for calculation


topp = np.column_stack((
    r*np.cos(vinkel), 
    r*np.sin(vinkel), 
    np.full(len(vinkel), h)
    )) #top of the cylinder

bunn = np.column_stack((
    r*np.cos(vinkel), 
    r*np.sin(vinkel), 
    np.zeros(len(vinkel))
    )) #bottom of the cylinder


pts = np.vstack((indre, skall, topp, bunn)) #combine all the points into one array

tri = Delaunay(pts) #create the tetrahedra

#indices of the points that make up the tetrahedra
p = pts[tri.simplices[:,0]] #first point of the tetrahedra
q = pts[tri.simplices[:,1]] #second point of the tetrahedra
r = pts[tri.simplices[:,2]] #third point of the tetrahedra
s = pts[tri.simplices[:,3]] #fourth point of the tetrahedra
f = 1 #formula is triple integral of 1 dV over an area of Omega

#three vectors from the same vertex
u = q - p
v = r - p
w = s - p

M = np.stack((u, v, w), axis = 1) #matrix with the three vectors as rows
dV = np.abs(1/6*np.linalg.det(M)) #1/6 from the formula of the volume for the tetrahedra

svar = np.sum(dV*f)
print(svar)
