# -*- coding: utf-8 -*-

import numpy as np


def DecompositionGS(A):
    """ Calcul de la décomposition QR de A une matrice carrée.
    L'algorithme de Gram-Schmidt est utilisé.
    La fonction renvoit (Q,R) """
    n,m=A.shape
    if n !=m :
        raise Exception('Matrice non carrée')

    Q=np.zeros((n,n))
    R=np.zeros((n,n))
    for j in range(n):
        for i in range(j):
            R[i,j]=Q[:,i]@A[:,j]
        w=A[:,j]
        for k in range(j):
            w=w-R[k,j]*Q[:,k]
        norme=np.linalg.norm(w)
        if norme ==0:
            raise Exception('Matrice non inversible')
        R[j,j]=norme
        Q[:,j]=w/norme
    return Q,R

def ResolTriSup(T,b):
    """Résolution d'un système triangulaire supérieur carré
    Tx=b
    La fonction ne vérifie pas la cohérence des dimensions de T et b
    ni que T est triangulaire supérieure.
    La fonction rend x sous la forme du même format que b."""


    n,m=T.shape
    x=np.zeros(n)
    for i in range(n-1,-1,-1):
        S=T[i,i+1:]@x[i+1:]
        x[i]=(b[i]-S)/T[i,i]
    x=np.reshape(x,b.shape)
    return x

def ResolTriInf(T,b):
    """Résolution d'un système triangulaire inférieur carré
    Tx=b
    La fonction ne vérifie pas la cohérence des dimensions de T et b
    ni que T est triangulaire inférieure.
    La fonction rend x sous la forme du même format que b."""


    n,m=T.shape
    x=np.zeros(n)
    for i in range(n):
        S=T[i,:i]@x[:i]
        x[i]=(b[i]-S)/T[i,i]
    x=np.reshape(x,b.shape)
    return x

def Cholesky(A):
    """
    Fonction qui calcule L la matrice de la décomposition de
    Cholesky de A une matrice réelle symétrique définie positive
    (A=LL^T où L est triangulaire inférieure).
    La fonction ne vérifie pas que A est symétrique.
    La fonction rend L.
    """

    n,m=A.shape
    if n != m:
        raise Exception('Matrice non carrée')
    L=np.zeros((n,n))
    for i in range(n):
        s=0.
        for j in range(i):
            s=s+L[i,j]**2
        R=A[i,i]-s
        if R<=0:
            raise Exception('Matrice non définie positive')
        L[i,i]=np.sqrt(R)
        for j in range(i+1,n):
            s=0.
            for k in range(i):
                s=s+L[i,k]*L[j,k]
            L[j,i]=(A[j,i]-s)/L[i,i]
    return L


