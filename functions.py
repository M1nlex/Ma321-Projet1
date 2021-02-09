import numpy as np
import matplotlib.pyplot as plt
from regression_lineaire import *

# Ajustement linéaire

# Tracer le nuage de points
# lecture des données 1
def open_data(file):
    p = open(file, "r")
    a = []
    for line in p:
        a.append(float(line))
    return a

# Lecture des données 2
p=np.loadtxt('dataP.dat')
q=np.loadtxt('dataQ.dat')

plt.scatter(p,q)

# Approx par une fonction
ComparaisonPolynomeRegression(p,q,1)
