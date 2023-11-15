# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# MRG packages
import _env
import preprocessing
import processing
import postprocessing
from utils import compute_projected, compute_gradient_descent2, compute_gradient_descent
#import solutions
from demo_control_polycopie2023 import your_optimization_procedure




# ----------------------------------------------------------------------
# -- Fell free to modify the function call in this cell.
# ----------------------------------------------------------------------
# -- set parameters of the geometry
N = 30 # number of points along x-axis
M = 2 * N  # number of points along y-axis
level = 1 # level of the fractal
spacestep = 1.0 / N  # mesh size


starting = []
final = []
frequencies = [1, 2, 3, 4]

for f in tqdm(frequencies) : 
    # -- set parameters of the partial differential equation
    wavenumber = 2*np.pi*f  # wavenumber
    epsilon1 = 10**(-5)
    epsilon2 = 10**(-1)
    beta = 0.1

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)



    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    Alpha = 10.0 - 10.0 * 1j
    # Alpha = 0 
    # -- this is the function you have written during your project
    #import compute_alpha
    #Alpha = compute_alpha.compute_alpha(...)
    alpha_rob = Alpha * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = np.sum(np.sum(chi)) / S  # constraint on the density
    V_obj = 0.5

    mu = 5  # initial gradient step

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    energy = np.zeros((100+1, 1), dtype=np.float64)
    ### WAVENUMBER  =  OMEGA
    chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                            Alpha, mu, chi, V_obj)
    # chi, energy, u, grad = solutions.optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                    Alpha, mu, chi, V_obj, mu1, V_0)
    # --- en of optimization

    chin = chi.copy()
    un = u.copy()

    # -- plot chi, u, and energy
    # postprocessing._plot_uncontroled_solution(u0, chi0)
    # postprocessing._plot_controled_solution(un, chin)
    # err = un - u0
    # postprocessing._plot_error(err)
    # postprocessing._plot_energy_history(energy)

    print(energy)
    starting.append(energy[0])
    final.append(energy[-1])


plt.plot(frequencies, starting, label = "starting energy")
plt.show()

