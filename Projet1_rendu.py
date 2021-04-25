"""
Noms: Blin, Dubois, El Khalil, Ignaczuk, Stevanovic
Prenoms: Marianne, Romaric, Ali, Elena, Alexandre
Classes: SC1 & SC2
"""

import numpy as np
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random as rdm
from methode_moindres_carre import *


# --------------------------------------
# Graphique page de garde
# --------------------------------------
def ImagePageDeGarde():
    fig = plt.figure(figsize=(8, 6))
    ax3d = plt.axes(projection="3d")

    xdata = np.linspace(-100, 100, 1000)
    ydata = np.linspace(-100, 100, 1000)
    X, Y = np.meshgrid(xdata, ydata)
    Z = X ** 2 + Y ** 2 - X * Y - X - Y
    # Z = np.sin(X)*np.cos(Y)

    ax3d = plt.axes(projection='3d')
    ax3d.plot_surface(X, Y, Z, cmap='plasma')
    ax3d.set_title('X**2 + Y**2 -X*Y - X - Y')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')

    plt.show()


# --------------------------------------
# 2.1.2.1
# --------------------------------------
def recup_donnees():
    p = np.loadtxt("./dataP.dat")  # extraction et mise dans une variable des donnees
    q = np.loadtxt("./dataQ.dat")  # extraction et mise dans une variable des donnees
    return (p, q)


# --------------------------------------
# 2.1.2.1 (bonus)
# --------------------------------------
def representation():
    [p, q] = recup_donnees()
    plt.scatter(p, q)  # representation du nuage de point des deux series de donnees
    plt.xlabel("L age des enfants (en annee)")
    plt.ylabel("La hauteur des enfants (en m)")
    # affichage du graphique
    plt.show()


# --------------------------------------
# 2.1.2.5
# --------------------------------------
def Creation_de_X(p):
    n = len(p)
    un = np.ones((n, 1))
    X = np.concatenate((un, np.transpose([p])), axis=1)
    return X


def Det_A(p):
    X = Creation_de_X(p)
    XT = np.transpose(X)
    XTX = np.dot(XT, X)
    return (XTX)


def Det_b(p, q):
    XT = np.transpose(Creation_de_X(p))
    XTq = np.dot(XT, q)
    return (XTq)


# --------------------------------------
# 2.2.1.1.a
# --------------------------------------
def val_vect_propres(p):
    XTX = Det_A(p)
    Val, Vec = np.linalg.eig(XTX)
    return (Val, Vec)


# --------------------------------------
# 2.2.1.1.b
# --------------------------------------
def conditionnement():
    lM = max(abs(val_vect_propres()[0]))
    lm = min(abs(val_vect_propres()[0]))
    return (lM / lm)


def diffcond(cond1, cond2):
    # on considère que le conditionnement calcule par numpy est plus proche de
    # la réalité
    diff = (abs(cond1 - cond2) / cond2) * 100
    return (diff)


# --------------------------------------
# 2.2.1.3.f
# --------------------------------------
def minimum_de_F(A, b, q):
    x = np.linalg.solve(A, b)
    k = 0.5 * np.dot(x, np.dot(A, np.transpose([x])))
    l = np.dot(b, np.transpose([x]))
    m = 0.5 * np.linalg.norm(q) ** 2
    return k - l + m


def fonction_partielles(A, b):
    e1 = np.array([[0], [1]])
    e2 = np.array([[1], [0]])
    v1, v2 = val_vect_propres()[1]
    v1 = np.transpose([v1])
    v2 = np.transpose([v2])
    I = np.linspace(-10, 10, 10000)
    D = [e1, e2, v1, v2]
    i = 1
    for d in D:
        K = []
        for t in I:
            k = 0.5 * np.dot(np.transpose(d), np.dot(A, d)) * t * t
            m = minimum_de_F()[0]
            F = (k + m)[0]
            K.append(F)
        plt.plot(I, K)
        plt.xlabel("t")
        plt.ylabel("F(c*+td)")
        plt.title("Courbe de la fonction partielle en c* suivant" + str(d))
        plt.show()


# --------------------------------------
# 2.2.1.4.b
# --------------------------------------
def q_2_2_1_4():
    pas = 0.05

    x = np.arange(-10, 10.000001, pas)
    y = np.arange(-10, 10.000001, pas)
    c1, c2 = np.meshgrid(x, y)

    p, q = recup_donnees()
    X = Creation_de_X(p)
    Z = X.T @ X
    s = q.T @ q
    w = X.T @ q

    F = 0.5 * (Z[0][0] * (c1 ** 2) + 2 * Z[0][1] * c1 * c2 + Z[1][1] * (c2 ** 2) - 2 * (w[0] * c1 + w[1] * c2) + s)

    res = plt.contour(c1, c2, F, 1000)
    plt.title('Courbes de niveau')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


# --------------------------------------
# 2.2.1.5.a
# --------------------------------------
def q_2_2_1_5():
    pas = 0.01
    x = np.arange(-10, 10.000001, pas)
    y = np.arange(-10, 10.000001, pas)
    c1, c2 = np.meshgrid(x, y)

    p, q = recup_donnees()
    X = Creation_de_X(p)
    Z = X.T @ X
    s = q.T @ q
    w = X.T @ q

    F = 0.5 * (Z[0][0] * (c1 ** 2) + 2 * Z[0][1] * c1 * c2 + Z[1][1] * (c2 ** 2) - 2 * (w[0] * c1 + w[1] * c2) + s)
    ax3d = plt.axes(projection='3d')
    ax3d.plot_surface(c1, c2, F, cmap='viridis', edgecolor='none')
    plt.title('Tracé de la fonction F')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


# --------------------------------------
# 3.2.2.4.a
# --------------------------------------
def GradientPasFixe(A, b, x0, rho, tol):
    nit = 1
    iMax = 5 * 10 ** 4
    xit = []
    r = 0
    sol = x0
    r = np.dot(A, x0) - np.transpose([b])
    d = -r
    sol = sol + rho * d
    xit.append(sol)
    nit = nit + 1
    while (np.linalg.norm(r) > tol and nit < iMax):
        r = np.dot(A, sol) - np.transpose([b])
        d = -r
        sol = sol + rho * d
        xit.append(sol)
        nit = nit + 1
    return (sol, xit, nit)


# --------------------------------------
# 3.2.3.6.a
# --------------------------------------
def GradientPasOptimal(A, b, x0, tol):
    itmax = 5 * 10 ** 4
    nit = 0
    xit = [x0]
    sol = x0
    xit.append(sol)
    r = np.dot(A, x0) - np.transpose([b])
    while (nit < itmax and np.linalg.norm(r) > tol):
        a = np.linalg.norm(r) ** 2 / np.dot(np.transpose(r), np.dot(A, r))[0][0]
        sol = sol - a * r
        r = np.dot(A, sol) - np.transpose([b])
        nit += 1
        xit.append(sol)
    return (sol, xit, nit)


# --------------------------------------
# 3.2.3.6.c
# --------------------------------------
def courbenombreiteration(A, b, x0):
    N = []
    I = []
    for k in range(1, 13):
        N.append(GradientPasOptimal(A, b, x0, 10 ** -k)[2])
        I.append(k)
    plt.plot(I, N)
    plt.xlabel("Tolérance (10^)")
    plt.ylabel("Nombre d'itération")
    plt.title("Nombre d'itération en fonction de la tolérance")
    plt.show()


# --------------------------------------
# 3.3.2.5
# --------------------------------------
def GradientConjugue(A, b, x0, e):
    itmax = 5 * 10 ** 4
    nit = 0
    xit = [x0]
    sol = x0
    r = np.dot(A, sol) - np.transpose([b])

    while nit < itmax and np.linalg.norm(r) > e:

        r = np.dot(A, sol) - np.transpose([b])

        if nit == 0:
            d = -r
        else:
            beta = np.linalg.norm(r) ** 2 / np.linalg.norm(r0) ** 2
            d = -r + beta * d

        rho = np.dot(r.T, r) / np.dot(np.dot(d.T, A), d)
        sol = sol + rho * d
        nit += 1
        r0 = r

        xit.append(sol)
    return sol, xit, nit


# --------------------------------------
# 3.4.1.1
# --------------------------------------
def q_3_4_1_1():
    # 3.4.1.1.a
    pas = 0.01
    x = np.arange(-10, 10.000001, pas)
    y = np.arange(-10, 10.000001, pas)
    c1, c2 = np.meshgrid(x, y)

    # 3.4.1.1.b
    c0 = np.array([[-9], [-7]])
    p, q = recup_donnees()
    X = Creation_de_X(p)
    Z = X.T @ X
    s = q.T @ q
    w = X.T @ q

    F = 0.5 * (Z[0][0] * (c1 ** 2) + 2 * Z[0][1] * c1 * c2 + Z[1][1] * (c2 ** 2) - 2 * (w[0] * c1 + w[1] * c2) + s)

    res = plt.contour(c1, c2, F, 1000)
    plt.title('Gradient des courbes de niveau et évolution de la solution')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 3.4.1.1.c
    sol1, xit1, nit1 = GradientPasOptimal(Det_A(p), Det_b(p, q), c0, 10 ** (-5))
    tempo_list11 = []
    tempo_list12 = []
    for i in xit1:
        tempo_list11.append(i[0][0])
        tempo_list12.append(i[1][0])
    plt.plot(tempo_list11, tempo_list12, linewidth=1.0, color='red')

    plt.show()


# --------------------------------------
# 3.4.1.2
# --------------------------------------
def q_3_4_1_2():
    c0 = np.array([[-9], [-7]])
    p, q = recup_donnees()

    # méthode 1
    sol1, xit1, nit1 = GradientPasOptimal(Det_A(p), Det_b(p, q), c0, 10 ** (-5))
    tempo_list11 = []
    tempo_list12 = []
    for i in xit1:
        tempo_list11.append(i[0][0])
        tempo_list12.append(i[1][0])
    plt.plot(tempo_list11, tempo_list12, linewidth=1.0, color='red', label='Pas opti (' + str(nit1) + ' itérations)')

    # méthode 2
    sol2, xit2, nit2 = GradientPasFixe(Det_A(p), Det_b(p, q), c0, 10 ** (-3), 10 ** (-6))
    tempo_list21 = []
    tempo_list22 = []
    for i in xit2:
        tempo_list21.append(i[0])
        tempo_list22.append(i[1])
    plt.plot(tempo_list21, tempo_list22, linewidth=1.0, color='blue', label='Pas fixe (' + str(nit2) + ' itérations)')

    # méthode 3
    sol3, xit3, nit3 = GradientConjugue(Det_A(p), Det_b(p,q), c0, 10**(-5))
    tempo_list31 = []
    tempo_list32 = []
    for i in xit3:
        tempo_list31.append(i[0])
        tempo_list32.append(i[1])
    plt.plot(tempo_list31, tempo_list32, linewidth=1.0, color='green', label='Conjugue (' + str(nit3) + ' itérations)')

    plt.legend(loc="lower right")
    plt.title('Evolution de la solution à chaque itération')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    p,q=recup_donnees()
    A = Det_A(p)
    b = Det_b(p, q)
    #x0 = np.array([[0],[0]])
    x0 = np.array([[-9],[-7]])
    tol = 10**(-20)
    sol, xit, nit = GradientPasOptimal(A, b, x0, tol)
    #print(sol)
    #print(xit)
    #print(nit)
    listeX = []
    listeY = []
    for i in xit:
        listeX.append(i[0])
        listeY.append(i[1])
    #print(listeX)
    #print(listeY)

    """
    axes = plt.gca()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    """
    plt.plot(listeX,listeY)
    plt.show()
