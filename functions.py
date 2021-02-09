import numpy as np
import matplotlib.pyplot as plt


def open_data(file):
    p = open(file, "r")
    a = np.zeros(50)
    for i in range(50):
        a[i] = p.readline()

    return a


