from Projet1_rendu import *
import time


def get_variables():
    p,q=recup_donnees()
    A = Det_A(p)
    b = Det_b(p, q)
    x0 = np.array([[-9],[-7]])
    tol = 10**(-10)
    rho = 10 ** (-3)
    solution = 0
    return p, q, A, b, x0, tol, rho, solution

def comparaison_vitesse_convergence():

    # Récupération des variables du problème
    p , q , A , b , x0 , tol, rho, solution = get_variables()

    # Récupération des solutions par itérations
    sol1, xit1, nit1 = GradientPasFixe(A, b, x0, rho, tol) # méthode du gradient à pas fixe
    sol2, xit2, nit2 = GradientPasOptimal(A, b, x0, tol) # méthode du gradient à pas optimal
    sol3, xit3, nit3 = GradientConjugue(A, b, x0, tol) # méthode du gradient conjugué

    vit1 = []
    vit11 = []
    vit2 = []
    vit3 = []

    # Listes des vitesses de convergence par itérations
    for i in range (1,len(xit1)):
        vit1.append( abs( xit1[i][1][0] - solution )/abs( xit1[i-1][1][0] - solution ) )

    for i in range (1,20):
        vit11.append( abs( xit1[i][1][0] - solution )/abs( xit1[i-1][1][0] - solution ) )

    for i in range (1,len(xit2)):
        vit2.append( abs( xit2[i][1][0] - solution )/abs( xit2[i-1][1][0] - solution ) )

    for i in range (1,len(xit3)):
        vit3.append( abs( xit3[i][1][0] - solution )/abs( xit3[i-1][1][0] - solution ) )

    #Affichage des résultats
    plt.subplot(221)
    plt.plot(range(0,len(vit1)) , vit1)
    plt.xlabel("Itération")
    plt.ylabel("Vitesse de convergence")
    plt.title("Vitesse de convergence de la méthode du gradient à pas fixe")


    plt.subplot(222)
    plt.plot(range(0,len(vit11)) , vit11)
    plt.xlabel("Itération")
    plt.ylabel("Vitesse de convergence")
    plt.title("Vitesse de convergence de la méthode du gradient à pas fixe avec zoom")



    plt.subplot(223)
    plt.plot(range(0,len(vit2)) , vit2)
    plt.xlabel("Itération")
    plt.ylabel("Vitesse de convergence")
    plt.title("Vitesse de convergence de la méthode du gradient à pas optimal")


    plt.subplot(224)
    plt.plot(range(0,len(vit3)) , vit3)
    plt.xlabel("Itération")
    plt.ylabel("Vitesse de convergence")
    plt.title("Vitesse de convergence de la méthode du gradient conjugué")

    print("Solution des méthodes :")
    print("gradient à pas fixe : "+str(sol1))
    print("gradient à pas optimal : "+str(sol2))
    print("gradient conjugué : "+str(sol3))

    plt.show()

def comparaison_nombre_iterations():
    # Récupération des variables du problème
    p , q , A , b , x0 , tol, rho, solution = get_variables()

    # Récupération des solutions par itérations
    sol1, xit1, nit1 = GradientPasFixe(A, b, x0, rho, tol) # méthode du gradient à pas fixe
    sol2, xit2, nit2 = GradientPasOptimal(A, b, x0, tol) # méthode du gradient à pas optimal
    sol3, xit3, nit3 = GradientConjugue(A, b, x0, tol) # méthode du gradient conjugué

    # Affichage des résultats
    print("Itérations totales :")
    print("gradient à pas fixe : "+str(nit1))
    print("gradient à pas optimal : "+str(nit2))
    print("gradient conjugué : "+str(nit3))

def comparaison_temps_calcul():
    # Récupération des variables du problème
    p , q , A , b , x0 , tol, rho, solution = get_variables()


    timelist1 = []
    timelist2 = []
    timelist3 = []
    #precision = range(1,21,1)
    precision = [10,20,30,40]

    for i in precision:
        # Nouvelle tolérance/précision
        tol = 10**((-1)*i)

        # Récupération des solutions par itérations
        start_time1 = time.time()
        sol1, xit1, nit1 = GradientPasFixe(A, b, x0, rho, tol) # méthode du gradient à pas fixe
        end_time1 = time.time() - start_time1
        timelist1.append(end_time1)

        start_time2 = time.time()
        sol2, xit2, nit2 = GradientPasOptimal(A, b, x0, tol) # méthode du gradient à pas optimal
        end_time2 = time.time() - start_time2
        timelist2.append(end_time2)

        start_time3 = time.time()
        sol3, xit3, nit3 = GradientConjugue(A, b, x0, tol) # méthode du gradient conjugué
        end_time3 = time.time() - start_time3
        timelist3.append(end_time3)

    # Affichage des résultats
    plt.subplot(411)
    plt.plot(precision,timelist1)
    plt.xlabel("Degré de précision ( 10^n )")
    plt.ylabel("Temps de calcul")
    plt.title("Temps de calcul en fonction de la précision pour la méthode du gradient à pas fixe")


    plt.subplot(412)
    plt.plot(precision,timelist2)
    plt.xlabel("Degré de précision ( 10^n )")
    plt.ylabel("Temps de calcul")
    plt.title("Temps de calcul en fonction de la précision pour la méthode du gradient à pas optimal")


    plt.subplot(413)
    plt.plot(precision,timelist3)
    plt.xlabel("Degré de précision ( 10^n )")
    plt.ylabel("Temps de calcul")
    plt.title("Temps de calcul en fonction de la précision pour la méthode du gradient conjugué")

    plt.subplot(414)
    plt.plot(precision,timelist1, label='gradient à pas fixe')
    plt.plot(precision,timelist2, label='gradient à pas optimal')
    plt.plot(precision,timelist3 ,label='gradient conjugué')
    plt.xlabel("Degré de précision ( 10^n )")
    plt.ylabel("Temps de calcul")
    plt.title("Temps de calcul en fonction de la précision pour la méthode du gradient conjugué")
    plt.legend(loc="lower right")

    plt.show()

if __name__ == '__main__':
    comparaison_temps_calcul()
