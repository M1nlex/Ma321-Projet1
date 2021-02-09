import numpy as np
import matplotlib.pyplot as plt


def open_data(file):
    p = open(file, "r")
    a = np.zeros(50)
    for i in range(50):
        a[i] = p.readline()

    return a


if __name__ == '__main__':
    p = open_data("dataP.dat")
    q = open_data("dataQ.dat")

    print(p, q)
    plt.plot(p, q, "x")
    plt.show()
