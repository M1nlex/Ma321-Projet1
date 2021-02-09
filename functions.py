import numpy as np
import matplotlib.pyplot as plt
from regression_lineaire import *

# Ajustement linéaire

# Tracer le nuage de points
def open_data(file):
    p = open(file, "r")
    a = []
    for line in p:
        a.append(float(line))
    return a

p=np.loadtxt('dataP.dat')
q=np.loadtxt('dataQ.dat')

plt.scatter(p,q)

ComparaisonPolynomeRegression(p,q,1)
