from Projet1_rendu import *
#from scipy.signal import argrelextrema
import scipy.signal
facteur = 10**(17)


def test_sens(nb):
    if (nb > 0):
        return  1
    elif (nb == 0):
        return  0
    else:
        return  -1

def points_change_dir(listeX , listeY): # FONCTIONS PAS TERMINEE
    # Créeation des liste de sortie
    LX = []
    LY = []

    # Calcul de la direction initiale sur l'axe x
    calcx = listeX[0] - listeX[1]
    sensX = test_sens(calcX)

    # Calcul de la direction initiale sur l'axe y
    calcy = listeY[0] - listeY[1]
    sensY = test_sens(calcy)

    for i in range(0,len(listeX)):
        calcx = listeX[i] - listeX[i+1]
        newsensX = test_sens(calcx)
        if (sensX != newsensX):
            LX.append(i)


def max_et_min_locaux(x,y):
    # sort the data in x and rearrange y accordingly
    sortId = np.argsort(x)
    x = x[sortId]
    y = y[sortId]
    #print(x)
    #print(y)
    # this way the x-axis corresponds to the index of x
    plt.plot(x, y)
    plt.show()
    maxm = argrelextrema(y, np.greater)
    minm = argrelextrema(y, np.less)
    return maxm,minm

def fonction_test():
    p,q=recup_donnees()
    A = Det_A(p)
    b = Det_b(p, q)
    x0 = np.array([[0],[0]])
    tol = 10**(-20)

    sol, xit, nit = GradientPasOptimal(A, b, x0, tol)

    listeX = []
    listeY = []
    for i in xit:
        listeX.append(i[0])
        listeY.append(i[1])

    #print(listeX)
    #print(listeY)


    axes = plt.gca()
    axes.set_xlim(0, 0.8)
    axes.set_ylim(0, 0.4)

    plt.plot(listeX,listeY)
    plt.show()


def test_autre_fonction(vector):
    #vector = [0, 6, 25, 20, 15, 8, 15, 6, 0, 6, 0, -5, -15, -3, 4, 10, 8, 13, 8, 10, 3,1, 20, 7, 3, 0 ]
    #print('Detect peaks without any filters.')
    indexes = scipy.signal.find_peaks_cwt(vector, np.arange(1, 7), max_distances=np.arange(1, 7)*2)
    indexes = np.array(indexes) - 1
    #print('Peaks are: %s' % (indexes))
    return indexes


def max_min_finder(liste):
    #ordre_grandeur = abs(max(liste))-abs(min(liste))

    max = []
    min = []

    if( len(liste)>3 ):
        for i in range(1,len(liste)-1):
            if( liste[i-1]<=liste[i] and liste[i]>=liste[i+1] ):
                max.append(i)
            if( liste[i-1]>=liste[i] and liste[i]<=liste[i+1] ):
                min.append(i)
        return(max,min)

def prog_final():
    p,q=recup_donnees()
    A = Det_A(p)
    b = Det_b(p, q)
    x0 = np.array([[0],[0]])
    tol = 10**(-20)

    sol, xit, nit = GradientPasOptimal(A, b, x0, tol)

    listeX = []
    listeY = []
    for i in xit:
        listeX.append( i[0][0] )
        listeY.append( i[1][0] )


    listeX = np.array(listeX)
    listeY = np.array(listeY)
    #print(listeX)
    print(listeY)
    plt.plot(listeX,listeY)
    #----------------------------------------------------------------
    maxi, mini = max_min_finder(listeY)
    print(maxi)
    print(mini)

    """
    for i in maxi:
        plt.scatter(listeX[i],listeX[i])
    for i in mini:
        plt.scatter(listeX[i],listeY[i])
    """
    pointA = [ listeX[maxi[0]] , listeY[maxi[0]] ]
    pointB = [ listeX[maxi[-1]] , listeY[maxi[-1]] ]
    pointC = [ listeX[mini[0]] , listeY[mini[0]] ]
    pointD = [ listeX[mini[-1]] , listeY[mini[-1]] ]
    print(pointA, pointB, pointC, pointD)
    plt.scatter( pointA[0], pointA[1] )
    plt.scatter( pointB[0], pointB[1] )
    plt.scatter( pointC[0], pointC[1] )
    plt.scatter( pointD[0], pointD[1] )


    plt.plot( [pointA[0], pointB[0]] , [pointA[1],pointB[1]] , 'r--', lw=1) # Red straight line
    plt.plot( [pointC[0], pointD[0]] , [pointC[1],pointD[1]] , 'r--', lw=1) # Red straight line
    plt.axis("equal")

    plt.show()

if __name__ == '__main__':
    prog_final()
