import numpy as np
from math import *
import matplotlib.pyplot as plt
from Projet1 import *
from mpl_toolkits import mplot3d

# --------------------------------------
# 3.4.1.1
# --------------------------------------

pas = 0.5

    #3.4.1.1.a
x = np.arange(-10,10.000001,pas)
y = np.arange(-10,10.000001,pas)
c1,c2 = np.meshgrid(x,y)
c0 = np.transpose(np.array([-9,-7]))

    #3.4.1.1.b
p,q = recup_donnees()
X = Creation_de_X(p)
Z = X.T@X
s = q.T@q
w=X.T@q

F = 0.5 * ( Z[0][0]*(c1**2) + 2*Z[0][1]*c1*c2 + Z[1][1]*(c2**2) - 2*(w[0]*c1 + w[1]*c2) + s )

#res = plt.contour(c1,c2,F,1000)

    # 3.4.1.1.c
sol1,xit1,nit1 = gradientPasOptimal(Det_A(p),Det_b(p,q),c0,10**(-5))
tempo_list11 = []
tempo_list12 = []
for i in xit1:
    tempo_list11.append( i[0] )
    tempo_list12.append( i[1] )
#plt.plot(tempo_list11, tempo_list12, linewidth=5.0, color='red')

#plt.show()

# --------------------------------------
# 3.4.1.2
# --------------------------------------

# méthode 1
plt.plot(tempo_list11, tempo_list12, linewidth=1.0, color='red', label='Pas opti')

# méthode 2
sol2,xit2,nit2 = GradientPasFixe(Det_A(p), Det_b(p,q), c0, 10**(-5), 10**(-5))
tempo_list21 = []
tempo_list22 = []
for i in xit2:
    tempo_list21.append( i[0] )
    tempo_list22.append( i[1] )
plt.plot(tempo_list21, tempo_list22, linewidth=1.0, color='blue', label='Pas fixe')


# méthode 3
sol3,xit3,nit3 = gradientConjugue(Det_A(p),Det_b(p,q),c0,10**(-5))
tempo_list31 = []
tempo_list32 = []
for i in xit3:
    tempo_list31.append( i[0] )
    tempo_list32.append( i[1] )
plt.plot(tempo_list31, tempo_list32, linewidth=1.0, color='green', label='Conjugue')

plt.legend(loc="lower right")
plt.show()

# --------------------------------------
# 3.4.1.3
# --------------------------------------

print("----------")
print("pas opti : "+str(nit1))
print("pas fixe : "+str(nit2))
print("conjugue : "+str(nit3))
print("----------")

# --------------------------------------
# 3.4.2
# --------------------------------------
print("Les méthodes convergent toutes, mais le nombre d'itérations varie fortement pour le gradient pas fixe.")
print("Cela est sans doute dû à l'estimation de départ.")

# 2.2.1.5 a
"""
ax3d = plt.axes(projection='3d')
ax3d.plot_surface(c1, c2, F)
plt.show()
"""
