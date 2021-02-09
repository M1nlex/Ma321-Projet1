import numpy as np
import matplotlib.pyplot as plt

# Ajustement linéaire

# Tracer le nuage de points V1
p = open("dataP.dat", "r")
q = open("dataQ.dat", "r")
a = np.zeros(50)
b = np.zeros(50)
for i in range(50):
    a[i] = p.readline()
    b[i] = q.readline()

print(a, b)

plt.plot(a,b, "x")
plt.show()

#Tracer le nuage de points V2
