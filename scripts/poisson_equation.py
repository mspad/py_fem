import numpy as np
import matplotlib.pyplot as plt
from py_fem.core.mesh import Mesh
from py_fem.utilities.solvers import poisson_solver
from py_fem.utilities.plot import plot_solution_2d
from py_fem.utilities.error_norm import compute_L2_error
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('app.log'),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    f = lambda x, y: 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    # u_analytical = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    u_analytical = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y) + x + y

    error = []
    n_ref = [10, 15, 25, 40]
    h = [1 / n for n in n_ref]
    h_squared = [1 / n ** 2 for n in n_ref]

    for n in n_ref:
        mesh = Mesh.unit_square(nx = n, ny = n, element_type = "P1")
        uh, A, F = poisson_solver(mesh=mesh,
                                  forcing_term_expression=f,
                                  dirichlet_bcs_expression=u_analytical,
                                  display_info=True)
        x = mesh.nodes[:, 0]
        y = mesh.nodes[:, 1]
        u_exact = u_analytical(x, y)
        error.append(compute_L2_error(mesh, uh, u_exact))
        logger.info(f"L2 norm of the error: {error[-1]}")
   
    
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


