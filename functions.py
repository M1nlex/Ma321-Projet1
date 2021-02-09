import numpy as np
import matplotlib.pyplot as plt


def open_data(file, length):
    p = open(file, "r")
    a = np.zeros(length)
    for i in range(length):
        a[i] = p.readline()

    return a

p=np.loadtxt('dataP.dat')
q=np.loadtxt('dataQ.dat')

plt.scatter(p,q)

