from Projet1_rendu import *

def show_graph1():
    B1 = 1
    B2 = 2
    a1 = 2
    a2 = 3
    angle = pi/3

    listeB1x = []
    listeB1y = []

    listeB2x = []
    listeB2y = []

    listex = np.arange(0,1.1,1)
    listey = []

    for j in listex:
        listey.append(np.sqrt(a2/a1)*np.tan(angle)*j+np.sqrt(B2)*np.sin(angle)*( (1/np.sqrt(a2))-(np.sqrt(a2)/a1) ))


    for i in np.arange(0,2*pi+pi/24,pi/24):
        listeB1x.append(np.sqrt(B1/a1)*np.cos(i))
        listeB1y.append(np.sqrt(B1/a2)*np.sin(i))
        listeB2x.append(np.sqrt(B2/a1)*np.cos(i))
        listeB2y.append(np.sqrt(B2/a2)*np.sin(i))

    # Ellipses
    plt.plot(listeB1x , listeB1y, label='ellipse beta=1')
    plt.plot(listeB2x , listeB2y, label='ellipse beta=2')

    # Points
    plt.scatter(0,0,c='red') # centre
    #plt.scatter(np.sqrt(B1/a1)*np.cos(angle) , np.sqrt(B1/a2)*np.sin(angle) , c = 'green') # intersection courbe 1
    plt.scatter(np.sqrt(B2/a1)*np.cos(angle) , np.sqrt(B2/a2)*np.sin(angle) , c = 'green', label='M(pi/3)') # intersection courbe 2

    # Droites
    plt.plot(listex,listey)

    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title("Courbe de niveau et droite D")
    plt.legend(loc="upper left")
    plt.axis('equal')
    plt.show()

def show_graph2():
    B1 = 1
    B2 = 2
    a1 = 2
    a2 = 3
    angle = pi/12

    listeB1x = []
    listeB1y = []

    listeB2x = []
    listeB2y = []

    listex = np.arange(0,1.1,1)
    listey = []

    for j in listex:
        listey.append(np.sqrt(a1/a2)*np.tan(angle)*j)


    for i in np.arange(0,2*pi+pi/24,pi/24):
        listeB1x.append(np.sqrt(B1/a1)*np.cos(i))
        listeB1y.append(np.sqrt(B1/a2)*np.sin(i))
        listeB2x.append(np.sqrt(B2/a1)*np.cos(i))
        listeB2y.append(np.sqrt(B2/a2)*np.sin(i))

    # Ellipses
    plt.plot(listeB1x , listeB1y, label='ellipse beta=1')
    plt.plot(listeB2x , listeB2y, label='ellipse beta=2')

    # Points
    plt.scatter(0,0,c='red') # centre
    #plt.scatter(np.sqrt(B1/a1)*np.cos(angle) , np.sqrt(B1/a2)*np.sin(angle) , c = 'green') # intersection courbe 1
    plt.scatter(np.sqrt(B2/a1)*np.cos(angle) , np.sqrt(B2/a2)*np.sin(angle) , c = 'green', label='M(pi/12)') # intersection courbe 2

    # Droites
    plt.plot(listex,listey)

    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title("Courbe de niveau et droite D_A")
    plt.legend(loc="upper left")
    plt.axis('equal')

    plt.show()

if __name__ == '__main__':
    show_graph1()
