import numpy as np
from math import *
import matplotlib.pyplot as plt
from Projet1 import *
from mpl_toolkits import mplot3d

met = 0
pas = 0.01
#3.4.1.a
x = np.arange(-10,10.000001,pas)
y = np.arange(-10,10.000001,pas)
c1,c2 = np.meshgrid(x,y)

#3.4.1.b
p,q = recup_donnees()

X = Creation_de_X(p)

Z = X.T@X
s = q.T@q
w=X.T@q

F = 0.5 * ( Z[0][0]*(c1**2) + 2*Z[0][1]*c1*c2 + Z[1][1]*(c2**2) - 2*(w[0]*c1 + w[1]*c2) + s )




if met == 0:
    print(c1)
    print(c2)
if met ==1: # méthode 1
    res = plt.contour(c1,c2,F,1000)
    plt.show()



# 2.2.1.5 a

ax3d = plt.axes(projection='3d')
ax3d.plot_surface(c1, c2, F)
plt.show()