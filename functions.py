import numpy as np
import matplotlib.pyplot as plt
from regression_lineaire import *

# Ajustement linéaire

# Lecture des données V1
def open_data(file, length):
    p = open(file, "r")
    a = np.zeros(length)
    for i in range(length):
        a[i] = p.readline()
    return a

# Lecture des données V2 et tracé du nuage de points
p = np.loadtxt('dataP.dat')
q = np.loadtxt('dataQ.dat')
plt.scatter(p,q)
#plt.show()

# Approximation par une fonction
ComparaisonPolynomeRegression(p,q,1)
