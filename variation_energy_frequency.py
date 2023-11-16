import numpy as np
import matplotlib.pyplot as plt
import demo_control_polycopie2023 as prog
from tqdm import tqdm

import processing
import preprocessing
import postprocessing
import _env

omega = np.linspace(100*2*np.pi ,200*2*np.pi, 50)

graph = []

if __name__ == "__main__":
    for i in tqdm(range(len(omega))):
        N = 70 # number of points along x-axis
        M = 2 * N  # number of points along y-axis
        level = 0   # level of the fractal
        spacestep = 1.0 / N  # mesh size

        # -- set parameters of the partial differential equation
        
        wavenumber = omega[i]
        epsilon1 = 10**(-2)
        epsilon2 = 10**(-1)
        beta = 0.01

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
        #alpha_rob = np.zeros(len(Alpha*chi))

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
        chi, energ, u, grad = prog.your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj)

        graph.append(energ[-1][0])
        print(graph)

print(np.array(graph))

f = 1 / (2 * np.pi) * omega

np.save('frequency.npy', f)
np.save('graph.npy', graph)

plt.plot(f, np.array(graph))
plt.show()