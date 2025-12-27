import numpy as np
import matplotlib.pyplot as plt
from py_fem.core.mesh import Mesh
from py_fem.core.boundary_conditions import apply_dirichlet_bcs
from py_fem.utilities.plot import plot_solution_2d
from py_fem.core.assemble import assemble_global_forcing, assemble_global_stiffness
from py_fem.utilities.error_norm import compute_L2_error
from scipy.sparse.linalg import cg

if __name__ == "__main__":
    f = lambda x, y: 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    # u_analytical = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    u_analytical = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y) + x + y

    error = []
    n_ref = [10, 15, 25, 40]
    h = [1 / n for n in n_ref]
    h_squared = [1 / n ** 2 for n in n_ref]

    for n in n_ref:
        print("-------------------------")
        mesh = Mesh.unit_square(nx = n, ny = n, element_type = "P1")
        print("Assembling lhs and rhs...")
        A = assemble_global_stiffness(mesh)
        print("A shape: ", A.get_shape(),)
        print("Nonzero entries over total entries:", A.count_nonzero(), "/", A.get_shape()[0] ** 2, "(", 100 * A.count_nonzero()/A.get_shape()[0] ** 2, "%)")
        F = assemble_global_forcing(mesh, f)
        dirichlet_bc = u_analytical(mesh.nodes[mesh.boundary_nodes][:, 0], mesh.nodes[mesh.boundary_nodes][:, 1])
        A, F = apply_dirichlet_bcs(A, F, mesh.boundary_nodes, dirichlet_bc)
        print("Solving the linear system Ax = F...")
        A.tocsr()
        uh, info = cg(A, F)
        if info == 0:
            print("Conjugate gradient successfully converged")
        else:
            print("Convergence not achieved")
        x = mesh.nodes[:, 0]
        y = mesh.nodes[:, 1]
        u_exact = u_analytical(x, y)
        error.append(compute_L2_error(mesh, uh, u_exact))
        print("L2 norm of the error =", error[-1])
        print("-------------------------")
    
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 8))
    axs[0, 0].loglog(h, error, '-o', label = "$L^2$ norm of the error")
    axs[0, 0].loglog(h, h_squared, '-o', label = "$O(h^2)$")
    axs[0, 0].legend()
    axs[0, 0].set_title("Convergence")

    plot_solution_2d(mesh, uh, show_mesh=True, ax = axs[0, 1])
    plot_solution_2d(mesh, abs(uh - u_analytical(mesh.nodes[:, 0], mesh.nodes[:, 1])), "Error", show_mesh=True, ax = axs[1, 1])

    axs[1, 0].spy(A, markersize=1)
    axs[1, 0].set_xlabel("Sparsity Pattern")
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    plt.show()
