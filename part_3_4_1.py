import numpy as np
from math import *
import matplotlib.pyplot as plt
from Projet1 import *
from mpl_toolkits import mplot3d

pas = 0.5

#3.4.1.a
x = np.arange(-10,10.000001,pas)
y = np.arange(-10,10.000001,pas)
c1,c2 = np.meshgrid(x,y)
c0 = np.transpose(np.array([-9,-7]))

#3.4.1.b
p,q = recup_donnees()
X = Creation_de_X(p)
Z = X.T@X
s = q.T@q
w=X.T@q

F = 0.5 * ( Z[0][0]*(c1**2) + 2*Z[0][1]*c1*c2 + Z[1][1]*(c2**2) - 2*(w[0]*c1 + w[1]*c2) + s )

res = plt.contour(c1,c2,F,1000)

# 3.4.1.c
sol,xit,nit = gradientPasOptimal(Det_A(p),Det_b(p,q),c0,10**(-5))
tempo_list1 = []
tempo_list2 = []
for i in xit:
    tempo_list1.append( i[0] )
    tempo_list2.append( i[1] )
plt.plot(tempo_list1,tempo_list2,linewidth=5.0,color='red')

plt.show()
