import numpy as np
# import dolfinx
# import ufl
# from mpi4py import MPI
# from petsc4py.PETSc import ScalarType
import matplotlib.pyplot as plt
# from dolfinx.nls.petsc import NewtonSolver
# from dolfinx.fem.petsc import NonlinearProblem
from scipy.integrate import solve_bvp
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
import scipy.io

# Set random seed for reproducibility
np.random.seed(42)

sample_num_to_test = 0 #choose from 0, 1, 2

# Define permeability function
def permeability(s):
    return 0.2 + s**2

# Define input function f(x)
def source_function(x):
    # return -2.5 * np.sin(np.pi * x) + 1.5 * np.sin(2 * np.pi * x) + 0.5 * x
    # return -1.98 * np.sin(6.02 * x - 0.14) - 0.48
    # return -1.4499*x**2+   1.4193*x +   0.0208
    if sample_num_to_test == 0:
        return -158.1375*x**6  + 522.8612*x**5 -643.3436*x**4 + 354.7193*x**3  -82.7021*x**2  +  5.5667*x  +  0.8322 # sample 0
    elif sample_num_to_test == 1:
        return  -138.9687*x**6 +  448.7337*x**5 -526.3932*x**4 +  275.1883*x**3  -64.5661*x**2+    5.4368*x+    0.1627 # sample 1
    elif sample_num_to_test == 2:
        return 175.3870*x**6 -489.5229*x**5+  482.3154*x**4 -200.1555*x**3+   34.6649*x**2   -3.6076*x+    0.5213 #sample 2


# FEM solver
# def solve_fem(nx=100, save=True):
#     mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, nx)
#     V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    
#     def boundary(x):
#         return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
    
#     bc = dolfinx.fem.dirichletbc(ScalarType(0), 
#                                 dolfinx.fem.locate_dofs_geometrical(V, boundary), V)
    
#     s = dolfinx.fem.Function(V)
#     v = ufl.TestFunction(V)
#     u = dolfinx.fem.Function(V)
#     x = mesh.geometry.x
#     u.interpolate(lambda x: source_function(x[0]))
        
#     # Define variational problem
#     F = (ufl.inner(permeability(s) * ufl.grad(s), ufl.grad(v)) * ufl.dx 
#         - u * v * ufl.dx)

#     # Create nonlinear problem
#     problem = NonlinearProblem(F, s, bcs=[bc])

#     # Create Newton solver
#     solver = NewtonSolver(MPI.COMM_WORLD, problem)

#     # Set solver parameters
#     solver.atol = 1e-8
#     solver.rtol = 1e-8
#     solver.max_it = 50

#     # Solve the nonlinear problem
#     n, converged = solver.solve(s)

#     if converged:
#         print(f"Newton solver converged in {n} iterations.")
#     else:
#         print("Newton solver did not converge.")
        
#     # Save solution to file
#     if save == True:
#         with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "darcy_solution.xdmf", "w") as file:
#             file.write_mesh(mesh)
#             file.write_function(s)

#     return x[:, 0], s.x.array, u.x.array

# Finite difference solver
def solve_fd(nx=100):
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]
    u = source_function(x)
    s = np.zeros(nx)
    
    for _ in range(10):
        kappa = permeability(s)
        main_diag = (kappa[1:] + kappa[:-1])/dx**2
        upper_diag = -kappa[1:-1]/dx**2
        lower_diag = -kappa[1:-1]/dx**2
        
        A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], 
                 shape=(nx-2, nx-2))
        A = csc_matrix(A)
        s_interior = spsolve(A, u[1:-1])
        s[1:-1] = s_interior
    
    return x, s, u

# solve with ODE solver from scipy
def solve_ssm(nx=100):
    def state_space_model(x, s):
        s1, s2 = s
        return np.vstack((s2, (-source_function(x)-2*s1*s2**2) / permeability(s1)))
    def bc(s_a, s_b):
        return np.array([s_a[0], s_b[0]])
    xs = np.linspace(0, 1, nx)
    s_init = np.zeros((2, xs.size))
    sol = solve_bvp(state_space_model, bc, xs, s_init)
    return sol.x, sol.y[0]

def fd_generate_example(nx=100, noise=None):
    '''nx gives the discretization and noise the amount of perturbation to source function, %.
    i.e. 0.01 corresponds to 1% noise.'''
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]
    u = source_function(x)
    s = np.zeros(nx)
    
    if noise is not None:
        assert type(noise)  == float, f"Input noise {noise} is not a float."
        meanvals = np.mean(np.abs(u), axis=1)
        rands = np.random.normal(loc=0, scale=noise, size=u.shape)

        additive_noise = rands * meanvals[:, np.newaxis]
        u = u + additive_noise
    
    for _ in range(10):
        kappa = permeability(s)
        main_diag = (kappa[1:] + kappa[:-1])/dx**2
        upper_diag = -kappa[1:-1]/dx**2
        lower_diag = -kappa[1:-1]/dx**2
        
        A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], 
                 shape=(nx-2, nx-2))
        A = csc_matrix(A)
        s_interior = spsolve(A, u[1:-1])
        s[1:-1] = s_interior
    if noise is not None:
        scipy.io.savemat(f"nonlineardarcy_test_noise_{noise}.mat", {"f_test": u, "u_test": s, "x": x})
        print(s.shape)
        print("Data with noise saved.")
    else:
        scipy.io.savemat(f"nonlineardarcy_train.mat", {"f_train": u, "u_train": s, "x": x})
        print(u.shape)
        print("Data without noise saved.")
    
    return None
   
# Solve and compare all 3 methods 
x_fd, s_fd, u_fd = solve_fd()
# x_fem, s_fem, u_fem = solve_fem()
x_ssm, s_ssm = solve_ssm()

fd_generate_example()

