# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 20:12:50 2021

@author: aliel
"""

import numpy as np
import matplotlib.pyplot as plt
from programme01 import *
import random as rdm
from math import *
from methode_moindres_carre import *



def RegressionLineaireEllipse(X,Y):     
    X=[X]
    Y=[Y]
    Xt=np.transpose(X)
    Yt=np.transpose(Y)
    n=np.shape(Xt)
    A=np.concatenate((Xt*Xt,Xt*Yt,Yt*Yt,Xt,Yt,np.ones(n)),axis=1)
    B=np.ones(n)
    M=ResolMCSVD(A,B)
    return M


def TraceEllipse(a,b,alpha,beta,teta):
    c=max(a,b)
    gamma=max(abs(alpha),abs(beta))
    x = -np.linspace(-c-gamma,c+gamma,1000)
    y = np.linspace(-c-gamma,c+gamma,1000)
    X,Y = np.meshgrid(x,y)
    A=(np.cos(teta)**2)/(a**2)+(np.sin(teta)**2)/(b**2)
    B=2*np.cos(teta)*np.sin(teta)
    C=(np.cos(teta)**2)/(b**2)+(np.sin(teta)**2)/(a**2)
    h=A+C
    D=-2*alpha*A-B*beta*h
    E=-2*beta*C-B*alpha*h
    F=(((alpha**2)*(np.cos(teta)**2)+(beta**2)*(np.sin(teta)**2))/(a**2))+(((beta**2)*(np.cos(teta)**2)+(alpha**2)*(np.sin(teta)**2))/(b**2))
    eqn = A*(X**2)+B*X*Y+C*(Y**2)+D*X+E*Y+F
    Z = 1
    plt.xlim([-c-gamma,c+gamma])
    plt.ylim([-c-gamma,c+gamma])
    plt.contour(X,Y,eqn,[Z])
    plt.grid()
    plt.show()


def TracerEllipseRegression(X,Y):
    M=RegressionLineaireEllipse(X,Y)
    A=M[0][0]
    B=M[1][0]
    C=M[2][0]
    D=M[3][0]
    E=M[4][0]
    F=M[5][0]
    teta=np.arcsin(B)/2
    b=np.sqrt((-(np.tan(teta)**2)*((np.sin(teta)**2)+(np.cos(teta)**2)))/C*(1-(A*np.tan(teta)**2)/C))
    a=np.sqrt(((b**2)*(np.cos(teta)**2))/((A*(b**2))-(np.sin(teta)**2)))
    h=A+C
    beta=(-2*E*A+B*D)/(2*A*(2*C+((h**2)*B**2)/(-2*A)))
    alpha=(D+B*beta*h)/(-2*A)
    p = -np.linspace(min(X)-1,max(X)+1,1000)
    q = np.linspace(min(Y)-1,max(Y)+1,1000)
    x,y = np.meshgrid(p,q)
    eqn = A*(x**2)+B*y*x+C*(y**2)+D*x+E*y+F
    Z = 1
    plt.scatter(X,Y)
    plt.scatter(alpha,beta)
    plt.contour(x,y,eqn,[Z])
    plt.show()


def RegressionLineairePolynomiale(X,Y,p):
    j=np.shape(X)[0]
    if np.shape(X)==(j,):
        X=[X]
    j=np.shape(Y)[0]
    if np.shape(Y)==(j,):
        Y=[Y]
    Xt=np.transpose(X)
    Yt=np.transpose(Y)
    M=np.ones(np.shape(Xt))
    F=np.ones(np.shape(Xt))
    for i in range(1,p+1):
        F=F*Xt
        M=np.concatenate((M,F),axis=1)
    return ResolMCNP(M,Yt)

def ComparaisonPolynomeRegression(X,Y,p):
    x=np.linspace(min(X),max(X),10000)
    y=[]
    R=RegressionLineairePolynomiale(X, Y, p)
    for i in x:
        P=R[0][0]
        for j in range (1,p+1):
            P=P+R[j][0]*i**j
        y.append(P)
    plt.plot(x,y)
    plt.scatter(X,Y)
    plt.show()

def RegressionLineaireCercle(X,Y):
    """

    Parameters
    ----------
    X : List
        Coordonnees en abscisse des points a etudier.
    Y : List
        Coordonnees en ordonee des points a etudier.

    Returns
    -------
    E : list
        liste contenant la coordonnee en abscisse et en ordonnee du
        centre et rayon du cercle de regression lineaire.

    """
    j=np.shape(X)[0]
    if np.shape(X)==(j,):
        X=[X]
    j=np.shape(Y)[0]
    if np.shape(Y)==(j,):
        Y=[Y]
    Xt=np.transpose(X)
    Yt=np.transpose(Y)
    n=np.shape(Xt)
    A=2*np.concatenate((Xt,Yt,1/2*np.ones(n)),axis=1)
    B=Xt*Xt+Yt*Yt
    M=ResolMCSVD(A,B)
    [a,b,g]=M
    r=np.sqrt(g+a**2+b**2)
    E=[a,b,r]
    return E

def TraceCercle(a,b,r,plot=False):
    """
    Fonction qui trace un cercle, avec un centre et un rayon donnés.

    Parameters
    ----------
    a : float
        Coordonnée en abscisse du centre du cercle.
    b : float
        Coordonnée en ordonnée du centre du cercle.
    r : float
        Rayon du cercle.

    Returns
    -------
    None.

    """
    theta=np.linspace(0,2*np.pi,1000)
    x=r*np.cos(theta)+a
    y=r*np.sin(theta)+b
    if plot==True:
        plt.plot(x,y)
        plt.axis('equal')
        plt.show()
    else :
        return x,y