import numpy as np
import matplotlib.pyplot as plt


def open_data():
    p = open("dataP.dat", "r")
    q = open("dataQ.dat", "r")
    a = np.zeros(50)
    b = np.zeros(50)
    for i in range(50):
        a[i] = p.readline()
        b[i] = q.readline()



if __name__ == '__main__':
    print(a, b)
    plt.plot(a,b, "x")
    plt.show()