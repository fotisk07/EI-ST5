import numpy 

def BelongsInteriorDomain(node):
	if (node < 0):
		return 1
	if node == 3:
		print("Robin")
		return 2
	else:
		return 0


def compute_gradient_descent(chi, grad, domain, mu):
	"""This function makes the gradient descent.
	This function has to be used before the 'Projected' function that will project
	the new element onto the admissible space.
	:param chi: density of absorption define everywhere in the domain
	:param grad: parametric gradient associated to the problem
	:param domain: domain of definition of the equations
	:param mu: step of the descent
	:type chi: numpy.array((M,N), dtype=float64
	:type grad: numpy.array((M,N), dtype=float64)
	:type domain: numpy.array((M,N), dtype=int64)
	:type mu: float
	:return chi:
	:rtype chi: numpy.array((M,N), dtype=float64

	.. warnings also: It is important that the conditions be expressed with an "if",
			not with an "elif", as some points are neighbours to multiple points
			of the Robin frontier.
	"""

	(M, N) = numpy.shape(domain)
	# for i in range(0, M):
	# 	for j in range(0, N):
	# 		if domain_omega[i, j] != _env.NODE_ROBIN:
	# 			chi[i, j] = chi[i, j] - mu * grad[i, j]
	# # for i in range(0, M):
	# 	for j in range(0, N):
	# 		if preprocessing.is_on_boundary(domain[i , j]) == 'BOUNDARY':
	# 			chi[i,j] = chi[i,j] - mu*grad[i,j]
	# print(domain,'jesuisla')
	#chi[50,:] = chi[50,:] - mu*grad[50,:]
	for i in range(1, M - 1):
		for j in range(1, N - 1):
			#print(i,j)
			#chi[i,j] = chi[i,j] - mu * grad[i,j]
			a = BelongsInteriorDomain(domain[i + 1, j])
			b = BelongsInteriorDomain(domain[i - 1, j])
			c = BelongsInteriorDomain(domain[i, j + 1])
			d = BelongsInteriorDomain(domain[i, j - 1])
			if a == 2:
				print(i+1,j, "-----", "i+1,j")
				chi[i + 1, j] = chi[i + 1, j] - mu * grad[i, j]
			if b == 2:
				print(i - 1, j, "-----", "i - 1, j")
				chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
			if c == 2:
				print(i, j + 1, "-----", "i , j + 1")
				chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
			if d == 2:
				print(i, j - 1, "-----", "i , j - 1")
				chi[i, j - 1] = chi[i, j - 1] - mu * grad[i, j]

	return chi


def P_l(chi , l, mu, J_prime, domain_omega):
    h,n=numpy.shape(chi)
    moy = numpy.mean(chi)
    chi_new=numpy.zeros((h,n))
    one=numpy.ones((h,n))
    zero=numpy.zeros((h,n))
    for i in range(numpy.shape(domain_omega)[0]):
        for j in range(numpy.shape(domain_omega)[1]):
            if domain_omega[i][j]==3:
                # print(i,j)
                moy_chi_new=0
                nb=1
                for colLoc in [-1,1]:
                    # moy_chi_new = 0
                    if (j+colLoc >= 0) and (j+colLoc < numpy.shape(domain_omega)[1]):
                        moy_chi_new += chi[i][j]-mu*J_prime[i][j+colLoc]+l
                        nb+=1
                moy_chi_new += chi[i][j]-mu*J_prime[i+1][j]+l
                moy_chi_new = moy_chi_new/nb
                chi_new[i,j]=max(0,min(moy_chi_new,1))
				
    return chi_new

def integrate(domain , var, step, boundary):
	integral = 0
	for i in range(len(var)):
		for j in range(len(var[0])):
			if domain[i,j] == boundary:
				integral = integral + var[i,j]*step**2
			

	return integral