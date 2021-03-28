"""
Nom:Ignaczuk
Prenom:Elena
Classe:C2 S
"""

import numpy as np
from math import *
import matplotlib.pyplot as plt

def recup_donnees():    #Partie 2, 2.1.2, question 1.
    p=np.loadtxt("./dataP.dat")#extraction et mise dans une variable des donnees
    q=np.loadtxt("./dataQ.dat")#extraction et mise dans une variable des donnees
    return(p,q)
def representation():   #bonus question précédente
    [p,q]=recup_donnees()
    plt.scatter(p,q)#representation du nuage de point des deux series de donnees
    plt.xlabel("L age des enfants (en annee)")
    plt.ylabel("La hauteur des enfants (en m)")
    plt.show()#affichage du graphique

p,q=recup_donnees()

def Creation_de_X(p):   #Partie 2, 2.1.2, question 5.
    n = len(p)
    un = np.ones((n, 1))
    X = np.concatenate((un, np.transpose([p])), axis=1)
    return X

def Det_A(p):           #Partie 2, 2.1.2, question 5.
    X=Creation_de_X(p)
    XT=np.transpose(X)
    XTX=np.dot(XT,X)
    return(XTX)

def Det_b(p,q):         #Partie 2, 2.1.2, question 5.
    XT=np.transpose(Creation_de_X(p))
    XTq=np.dot(XT,q)
    return(XTq)


def val_vect_propres(): #Partie 2, 2.2.1, question 1.a)
    XTX=Det_A(p)
    Val,Vec =np.linalg.eig(XTX)
    return(Val,Vec)

def conditionnement():  #Partie 2, 2.2.1, question 1.b)
    lM=max(abs(val_vect_propres()[0]))
    lm=min(abs(val_vect_propres()[0]))
    return(lM/lm)

def diffcond(cond1,cond2):      #Partie 2, 2.2.1, question 1.b)
    #on considère que le conditionnement calcule par numpy est plus proche de
    #la réalité
    diff=(abs(cond1-cond2)/cond2)*100
    return(diff)

def minimum_de_F():     #Partie 2, 2.2.1, question 3.f)  (on utilise cette fonction pour la fonction après, pour tracer les fonctions partielles)
    x=np.linalg.solve(A,b)
    k=0.5*np.dot(x, np.dot(A,np.transpose([x])))
    l=np.dot(b, np.transpose([x]))
    m=0.5*np.linalg.norm(q)**2
    return k-l+m


def fonction_partielles(): #Partie 2, 2.2.1, question 3.f)
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

def GradientPasFixe(A,b,x0,rho,tol):        #Partie 3, 3.2.2, question 4.a)
    nit=1
    iMax=5*10**4
    xit=[]
    r=0
    sol=x0
    r=np.dot(A,x0)-np.transpose([b])
    d=-r
    sol=sol+rho*d
    xit.append(sol)
    nit=nit+1
    while(np.linalg.norm(r)>tol and nit<iMax):
        r=np.dot(A,sol)-np.transpose([b])
        d=-r
        sol=sol+rho*d
        xit.append(sol)
        nit=nit+1
    return(sol,xit,nit)

"""
def gradientPasOptimal(A,b,x0,tol):
    nit=1
    iMax=5*10**4
    xit=[]
    r=np.dot(A,x0)-b
    sol=x0
    rT=np.transpose(r)
    d=-r
    rho=np.dot(rT,r)/np.dot(np.dot(rT,A),r)
    sol=sol+rho*d
    xit.append(sol)
    nit=nit+1
    while(np.linalg.norm(r)>tol and nit<iMax):
        r=np.dot(A,sol)-b
        rT=np.transpose(r)
        d=-r
        rho=np.dot(rT,r)/np.dot(np.dot(rT,A),r)
        sol=sol+rho*d
        xit.append(sol)
        nit=nit+1

    return(sol,xit,nit)
"""

A=Det_A(p)
b=Det_b(p, q)
x0=np.array([[-9],[-7]])

def GradientPasOptimal(A,b,x0,tol):     #Partie 3, 3.2.3, question 6.a)
    itmax=5*10**4
    nit=0
    xit=[x0]
    sol=x0
    xit.append(sol)
    r=np.dot(A,x0)-np.transpose([b])
    while (nit<itmax and np.linalg.norm(r)>tol):
        a=np.linalg.norm(r)**2/np.dot(np.transpose(r), np.dot(A,r))[0][0]
        sol=sol-a*r
        r=np.dot(A,sol)-np.transpose([b])
        nit+=1
        xit.append(sol)
    return (sol, xit, nit)

def courbenombreiteration():    #Partie 3, 3.2.3, question 6. c)
    N=[]
    I=[]
    for k in range(1,13):
        N.append(GradientPasOptimal(A,b,x0,10**-k)[2])
        I.append(k)
    plt.plot(I,N)
    plt.xlabel("Tolérance (10^)")
    plt.ylabel("Nombre d'itération")
    plt.title("Nombre d'itération en fonction de la tolérance")
    plt.show()

"""def gradientConjugue(A,b,x0,tol):
    nit=1
    iMax=5*10**5
    xit=[]
    R=[]
    D=[]
    RT=[]
    DT=[]
    r=np.dot(A,x0)-b
    R.append(r)
    sol=x0
    rT=np.transpose(r)
    RT.append(rT)
    d=-r
    D.append(d)
    dT=np.transpose(d)
    DT.append(dT)

    rho=np.dot(rT,r)/np.dot(np.dot(rT,A),r)
    sol=sol+rho*d
    xit.append(sol)
    nit=nit+1

    while(np.linalg.norm(r)>tol and nit<iMax):
        #print("nit:",nit)
        #iMax=5*10**4
        r=np.dot(A,sol)-b
        R.append(r)
        rT=np.transpose(r)
        RT.append(rT)
        beta=(np.linalg.norm(r))**2/(np.linalg.norm(R[nit-2]))**2
        d=-r+beta*D[nit-2]
        D.append(d)
        dT=np.transpose(d)
        DT.append(dT)
        rho=np.dot(rT,r)/np.dot(np.dot(dT,A),d)
        sol=sol+rho*d
        xit.append(sol)
        nit=nit+1
    return(sol,xit,nit)

"""

def GradientConjugue(A,b,x0, e):    #Partie 3, 3.3.2, question 5.
    itmax=5*10**4
    nit=0
    xit=[x0]
    sol=x0
    r=np.dot(A,x0)-np.transpose([b])
    d=-r
    while (nit<itmax and np.linalg.norm(r)>e):
        a=-np.dot(np.transpose(r),d)[0][0]/np.dot(np.transpose(d),np.dot(A,d))[0][0]
        sol=sol+a*d
        beta=np.dot(np.transpose(r), np.dot(A,d))[0][0]/np.dot(np.transpose(d),np.dot(A,d))[0][0]
        r=np.dot(A,sol)-np.transpose([b])
        d=-r+beta*d
        nit+=1
        xit.append(sol)
    return (sol,xit,nit)



if __name__=='__main__':
    #Programme principal
    print("2.1 Formulation et analyse mathematique")
    print("2.1.2 Ajustement lineair")

    [p,q]=recup_donnees()
    print(representation())


    print("2.2 Le probleme (Q) est-il bien poser ?")
    print("2.2.1 Autour de la fonction quadratique F")
    print("Question 1: Couples propres et conditionnements de XTX")
    print("a\n")

    Val,Vec = val_vect_propres()
    vect1=[row[0] for row in Vec]
    vect2=[row[1] for row in Vec]
    print("Le 1er couple valeur propre-vecteur propre de A est:\n(lambda1,v1)=\n","(",Val[0],",",vect1,")")
    print("Le 2ieme couple valeur propre-vecteur propre de A est:\n(lambda2,v2)=\n","(",Val[1],",",vect2,")")

    print("b\n")

    condcal=conditionnement()
    condnp=np.linalg.cond(Det_A(p))
    print("cond(XTX)=",conditionnement())
    print("le conditionnement selon numpy, cond(XTX)=",np.linalg.cond(Det_A(p)))
    print("L'écart relatif des deux conditionnements est de:",diffcond(condcal,condnp)) #plus particulierement, on ne considere qu une precision exacte qu a
    #10^-2 près

    print("Question 2: Quelques propriété élémentaires de F")
    print("a)\n")

    print(" Soit la fonction f : R^N → R quadratique definie sous la forme matricielle. La fonction f est dite definie positive si et seulement si toutes les valeurs propres de A sont strictement positives.D'apres Question1.a), on a lambda1=5.210865991628907 et lambda2=1403.0938641799348qui sont des valeurs positives donc F est quadratique définie positive")
    print("\n")

    print("b)\n")

    print("Soient f : R^N → R une fonction quadratique definie positive d expression matricielle. Alors f est coercive sur R^N et f est strictement convexe sur R^N")
    print("\n")

    print("c)\n")
    #a voir dans le cours magistral (correction leçon 1 exo2.5.1)

    print("\nQuestion 3: Fonctions partielles de F")
    print("\n3.2 Methode de descente du gradient ou de plus forte descente")
    print("\n3.2.1 Direction de plus forte descente")
    print("\n3.2.2 L’algorithme de descente du gradient a pas fixe")
    print("\n4)")

    sol,xit,nit=GradientPasFixe(Det_A(p),Det_b(p,q),np.transpose(np.array([-9,-7])),10**(-3),10**(-6))
    print("\n a) la solution est le vecteur :",sol,"elle appartient à la suite (ci)",xit,"on retrouve la solution au bout de",nit,"iterations")
    sola,xita,nita=GradientPasFixe(Det_A(p),Det_b(p,q),np.transpose(np.array([-9,-7])),10**(-1),10**(-6))
    print("\n b) la solution est le vecteur :",sola,"elle appartient à la suite (ci)",xita,"on retrouve la solution au bout de",nita,"iterations pour rho=10^(-1)")
    #pas trop grand: on sort plusieurs fois de la boucle, resultat non représentatif et resultat attendu non atteint
    solb,xitb,nitb=GradientPasFixe(Det_A(p),Det_b(p,q),np.transpose(np.array([-9,-7])),10**(-5),10**(-6))
    print("\n b) la solution est le vecteur :",solb,"elle appartient à la suite (ci)",xitb,"on retrouve la solution au bout de",nitb,"iterations pour rho=10^(-5)")
    #pas trop petit: le nombre d'iteration demande est bien trop grand 5000 iterations

    print("\n3.2.3 L’algorithme de descente du gradient a pas optimal")
    print("\n6)")

    sol1,xit1,nit1=gradientPasOptimal(Det_A(p),Det_b(p,q),np.transpose(np.array([-9,-7])),10**(-6))
    print("\n a)la solution est le vecteur :",sol1,"elle appartient à la suite (ci)",xit1,"on retrouve la solution au bout de",nit1,"iterations")
    #b)Calcul du pas pour que le nombre d'iteration soit minium donc moins long que la methode du gradient à pas fixe

    print("\n3.3.1 Direction du gradient conjugue")
    sol2,xit2,nit2=gradientConjugue(Det_A(p),Det_b(p,q),np.transpose(np.array([-9,-7])),10**(-6))
    print("\n a) la solution est le vecteur :",sol2,"elle appartient à la suite (ci)",xit2,"on retrouve la solution au bout de",nit2,"iterations")
    #Nombre d'iteration vraiment tres petit: la methode la mieux optimisee grace a la recherche du coefficient de conjugaison
