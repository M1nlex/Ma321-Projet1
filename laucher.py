import numpy as np
import matplotlib.pyplot as plt
from functions import *

# Lecture des données
#p = open_data('dataP.dat', 50)
#q = open_data('dataQ.dat', 50)
p = np.loadtxt('dataP.dat')
q = np.loadtxt('dataQ.dat')

# Nuage de points
plt.ylabel("Hauteur (m)")
plt.xlabel("Age (années)")
plt.scatter(p,q)
plt.title("Hauteur en fonction de l'âge \n", fontsize=12)
plt.show()
