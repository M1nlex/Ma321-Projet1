from programme01 import *
import numpy as np


def ResolMCEN(A,b):
    B = np.dot(A.T,b)
    L = Cholesky(np.dot(A.T,A))
    Lt = L.T #Transposée de L
    Y=ResolTriInf(L,B)
    X=ResolTriSup(Lt,Y)
    return X


def DecompositionGSGenerale(A):
    """

    Parameters
    ----------
    A : Matrice
        Matrice que l'on souhaite décomposer sous la forme QR.
        De format (n,p), on peut avoir n=p, n>p et n<p.

    Raises
    ------
    Exception
        Décomposition QR impossible si on n'a pas Ker(A)={0}.

    Returns
    -------
    Q : Matrice
        Q, de taille (n,p), vérifie Qt.Q=Ip, matrice identitée de taille p.
        Attention : Q.Qt n'est pas nécessairement égale à In.
    R : Matrice
        Matrice triangulaire supérieure de taille p.

    """

    n,p=np.shape(A)
    Q=np.zeros((n,p))
    R=np.zeros((p,p))
    for j in range (0,p):
        for i in range (0,j):
            R[i,j]=np.vdot(A[:,j],Q[:,i])
        w=A[:,j]
        for k in range(j):
            w=w-R[k,j]*Q[:,k]
        norme=np.linalg.norm(w)
        if norme ==0:
            raise Exception('décomposition QR impossible : Ker(A)!={0}')
        R[j,j]=norme
        Q[:,j]=(1/norme)*w
    return Q,R

def ResolMCQR(A,b):
    Q,R=DecompositionGSGenerale(A)
    Qt=Q.T
    x=ResolTriSup(R,np.dot(Qt,b))
    return x


def ResolMCNP(A,b):
    return np.linalg.lstsq(A,b,rcond=None)[0]

def ResolMCSVD(A,b):
    """
    Fonction qui résout le problème des moindres carrés en utilisant
    la décomposition en valeur singulière.

    Parameters
    ----------
    A : array
        Matrice A du système Ax=b.
    b : array
        Matrice b du système Ax=b.

    Returns
    -------
    x : array
        Solution au moindre carré par la déomposition en valeur
        singulière.

    """
    K=np.linalg.svd(A,False)
    S_p_i=np.diag(1/K[1])
    U_e=np.transpose(K[0])
    V=np.transpose(K[2])
    A_p_i=V@S_p_i@U_e
    x=np.dot(A_p_i,b)
    return x
