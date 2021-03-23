# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:20:38 2021

@author: aliel
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 08:26:23 2021

@author: aliel
"""

from functions import *
import numpy as np
import matplotlib.pyplot as plt
from Projet1 import *

p=np.loadtxt('dataP.dat')
q=np.loadtxt('dataQ.dat')


A=Det_A(p)
b=Det_b(p,q)

def minimum_de_F():         #fonction qui retourne le minimum de F pour l'utiliser dans le traçage des fonctions partielles au point critique suivant un vecteur
    x=np.linalg.solve(A,b)
    return 0.5*np.dot(np.transpose([x]), np.dot(A,x))-np.dot(np.transpose([b]), x)+0.5*np.linalg.norm(q)**2


def fonction_partielles(): #traçage des courbes des fonctions partielles au point critiques et selon les vecteurs de la base canonique(e1 et e2) et les vecteurs propres de A (v1 et v2)
    e1=[[0],[1]]
    e2=[[1],[0]]
    v1,v2=val_vect_propres()[1] #fonction dans le code d'Elena
    v1=np.transpose([v1])
    v2=np.transpose([v2])
    I=np.linspace(-10,10,10000)
    D=[e1,e2,v1,v2]
    for d in D:
        K=[]
        for t in I:
            F=0.5*np.dot(np.transpose(d),np.dot(A,d))*t*t+minimum_de_F()
            K.append(F)
    plt.plot(a,K)
    plt.show()

