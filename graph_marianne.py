from Projet1_rendu import *

def show_graph2_marianne():
    B0 = 0
    B1 = 1
    B2 = 2
    a1 = 2
    a2 = 3

    listeB1x = []
    listeB1y = []

    listeB2x = []
    listeB2y = []

    listeB0x = []
    listeB0y = []

    for i in np.arange(0,2*pi+pi/24,pi/24):
        listeB0x.append(np.sqrt(B0/a1)*np.cos(i))
        listeB0y.append(np.sqrt(B0/a2)*np.sin(i))
        listeB1x.append(np.sqrt(B1/a1)*np.cos(i))
        listeB1y.append(np.sqrt(B1/a2)*np.sin(i))
        listeB2x.append(np.sqrt(B2/a1)*np.cos(i))
        listeB2y.append(np.sqrt(B2/a2)*np.sin(i))

    # Ellipses
    plt.plot(listeB0x , listeB0y, label='ellipse beta=0')
    plt.plot(listeB1x , listeB1y, label='ellipse beta=1')
    plt.plot(listeB2x , listeB2y, label='ellipse beta=2')

    # Points
    plt.scatter(0,0,c='blue') # centre

    plt.axis('equal')
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title("Courbes de niveau")
    plt.legend(loc="upper left")

    plt.show()


def show_graph1_marianne():

    B0 = 0
    B1 = 1
    B2 = 2
    a1 = 2
    a2 = 3

    listeB1x = []
    listeB1y = []

    listeB2x = []
    listeB2y = []

    listeB0x = []
    listeB0y = []

    for i in np.arange(0,2*pi+pi/24,pi/24):
        listeB0x.append(np.sqrt(B0)*np.cos(i))
        listeB0y.append(np.sqrt(B0)*np.sin(i))
        listeB1x.append(np.sqrt(B1)*np.cos(i))
        listeB1y.append(np.sqrt(B1)*np.sin(i))
        listeB2x.append(np.sqrt(B2)*np.cos(i))
        listeB2y.append(np.sqrt(B2)*np.sin(i))

    # Ellipses
    plt.plot(listeB0x , listeB0y, label='ellipse beta=0')
    plt.plot(listeB1x , listeB1y, label='ellipse beta=1')
    plt.plot(listeB2x , listeB2y, label='ellipse beta=2')

    # Points
    plt.scatter(0,0,c='blue') # centre

    plt.axis('equal')
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title("Courbes de niveau")
    plt.legend(loc="upper left")

    plt.show()


if __name__ == '__main__':
    show_graph2_marianne()
