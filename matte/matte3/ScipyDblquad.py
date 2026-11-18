from scipy.integrate import dblquad
from math import sin

# a)
f = lambda x,y: x*y*y
Ia = dblquad(f, 0,1,2,3)

# b)
g = lambda y,x: sin(x**4+2*y**2)
Ib = dblquad(g, 0,1,-1,1)

# c)
Ic = dblquad(f, 0,1,0,lambda x:x)

# d)
d = lambda y,x: x*y*y
Id = dblquad(f, 0,1,0,lambda x: 1-x)

print(Ia[0],Ib[0],Ic[0],Id[0])