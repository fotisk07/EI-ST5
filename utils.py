import numpy as np
import _env
from processing import *


def compute_projected(chi, domain, V_obj):
    """This function performs the projection of $\chi^n - mu*grad

    To perform the optimization, we use a projected gradient algorithm. This
    function caracterizes the projection of chi onto the admissible space
    (the space of $L^{infty}$ function which volume is equal to $V_{obj}$ and whose
    values are located between 0 and 1).

    :param chi: density matrix
    :param domain: domain of definition of the equations
    :param V_obj: characterizes the volume constraint
    :type chi: numpy.array((M,N), dtype=float64)
    :type domain: numpy.array((M,N), dtype=complex128)
    :type float: float
    :return:
    :rtype:
    """

    (M, N) = numpy.shape(domain)
    S = 0
    for i in range(M):
        for j in range(N):
            if domain[i, j] == _env.NODE_ROBIN:
                S = S + 1

    B = chi.copy()
    l = 0
    chi = set2zero(chi, domain)

    V = numpy.sum(numpy.sum(chi)) / S
    debut = -numpy.max(chi)
    fin = numpy.max(chi)
    ecart = fin - debut
    # We use dichotomy to find a constant such that chi^{n+1}=max(0,min(chi^{n}+l,1)) is an element of the admissible space
    while ecart > 10 ** -4:
        # calcul du milieu
        l = (debut + fin) / 2
        for i in range(M):
            for j in range(N):
                chi[i, j] = numpy.maximum(0, numpy.minimum(B[i, j] + l, 1))
        chi = set2zero(chi, domain)
        V = sum(sum(chi)) / S
        if V > V_obj:
            fin = l
        else:
            debut = l
        ecart = fin - debut
        # print('le volume est', V, 'le volume objectif est', V_obj)

    return chi





def BelongsInteriorDomain(node):
	if (node < 0):
		return 1
	if node == 3:
		# print("Robin")
		return 2
	else:
		return 0

def gradient_descent_student(chi, grad, domain_omega,mu):
	(M, N) = np.shape(domain_omega)
	for i in range(0, M):
		for j in range(0, N):
			if domain_omega[i, j] == _env.NODE_ROBIN:
				chi[i,j] = chi[i,j] - mu*grad[i -1 ,j]
			
	return chi




def integrate(domain , var, step, boundary):



	integral = 0
	for i in range(len(var)):
		for j in range(len(var[0])):
			if domain[i,j] == boundary:
				integral = integral + var[i,j]*step**2
			

	return integral