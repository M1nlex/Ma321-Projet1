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




def minimum_de_F():         
    x=np.linalg.solve(A,b)
    k=0.5*np.dot(x, np.dot(A,np.transpose([x])))
    l=np.dot(b, np.transpose([x]))
    m=0.5*np.linalg.norm(q)**2
    return k-l+m


def fonction_partielles(): 
    e1=np.array([[0],[1]])
    e2=np.array([[1],[0]])
    v1,v2=val_vect_propres()[1] 
    v1=np.transpose([v1])
    v2=np.transpose([v2])
    I=np.linspace(-10,10,10000)
    D=[e1,e2,v1,v2]
    i=1
    for d in D:
        K=[]
        for t in I:
            k=0.5*np.dot(np.transpose(d),np.dot(A,d))*t*t
            m=minimum_de_F()[0]
            F=(k+m)[0]
            K.append(F)
        plt.plot(I,K)
        plt.xlabel("t")
        plt.ylabel("F(c*+td)")
        plt.title("Courbe de la fonction partielle en c* suivant"+str(d))
        plt.show()


def pointcritique():
    m=len(p)
    k=np.dot(p, np.ones(m))
    l=m*np.linalg.norm(p)**2-k**2
    K=((1/m)+k/(l*m))*np.dot(q,np.ones(m))
    L=(-k**2/(m*l))*np.dot(q,np.ones(m))+(k/l)*np.dot(p,np.transpose([q]))
    c=np.array([K,L[0]])
    return c

def Gradientconjugue2(x0, e): #Le programme ne donne pas de bon résultat
    A=Det_A(p)
    b=Det_b(p,q)
    itmax=5*10**4
    nite=0
    xit=[x0]
    xk=x0
    r=np.dot(A,x0)-np.transpose([b])
    d=-r
    while (nite<itmax and np.linalg.norm(r)>e):
        a=-np.dot(np.transpose(r),d)[0][0]/np.dot(np.transpose(d),np.dot(A,d))[0][0]
        xk=xk+a*d
        beta=np.dot(np.transpose(r), np.dot(A,d))[0][0]/np.dot(np.transpose(d),np.dot(A,d))[0][0]
        r=np.dot(A,xk)-np.transpose([b])
        d=-r+beta*d
        nite+=1 
        xit.append(xk)
    return (nite,xk,xit)

def GradientPasOptimal(A,b,x0,tol):
    itmax=5*10**4
    nite=0
    xit=[x0]
    xk=x0
    r=r=np.dot(A,x0)-np.transpose([b])
    while (nite<itmax and np.linalg.norm(r)>tol):
        a=np.linalg.norm(r)**2/np.dot(np.transpose(r), np.dot(A,r))[0][0]
        xk=xk-a*r
        r=np.dot(A,xk)-np.transpose([b])
        nite+=1
        xit.append(xk)
    return (sol, xit, nit)

