import numpy as np


def open_data(file):
    p = open(file, "r")
    a = []
    for line in p:
        a.append(float(line))

    return a


