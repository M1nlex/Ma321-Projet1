import numpy as np
import matplotlib.pyplot as plt
from regression_lineaire import *


def open_data(file):
    p = open(file, "r")
    a = []
    for line in p:
        a.append(float(line))

    return a


# Ajustement linéaire


# Tracer le nuage de points


ComparaisonPolynomeRegression(p,q,1)


