import numpy as np
from py_fem.core.mesh import Mesh
from typing import Callable
import logging
from py_fem.core.assemble import assemble_global_forcing, assemble_global_stiffness
from py_fem.core.boundary_conditions import apply_dirichlet_bcs
from scipy.sparse.linalg import cg
logger = logging.getLogger(__name__)

def poisson_solver(mesh: Mesh, 
                   forcing_term_expression: Callable[[float, float], float], 
                   dirichlet_bcs_expression: Callable[[float, float], float] | None = None,
                   display_info: bool = True
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 2D Poisson equation -div(grad(u)) = f

    Parameters
    ----------
    mesh : Mesh
        Mesh object
    forcing_term_expression : Callable[[float, float], float]
        forcing term as Callable function
    dirichlet_bcs_expression : Callable[[float, float], float] | None, optional
        dirichlet bc expression as Callable function, by default None
    display_info : bool, optional
        if True activate logging, by default True

    Returns
    -------
    np.ndarray
        numerical solution
    np.ndarray
        stiffness matrix
    np.ndarray
        forcing term
    """
    
    if not display_info:
        logger.setLevel("CRITICAL")
    logger.info("##########################################################")
    logger.info("Solving Poisson Equation: -div(grad(u)) = f")
    logger.info("Assembling stiffness matrix...")
    A = assemble_global_stiffness(mesh)
    logger.info(f"A shape: {A.get_shape()}")
    logger.info(f"Nonzero entries over total entries: {A.count_nonzero()}/{A.get_shape()[0] ** 2}({100 * A.count_nonzero()/A.get_shape()[0] ** 2}%)")
    logger.info("Assembling forcing term...")
    F = assemble_global_forcing(mesh, forcing_term_expression)
    if dirichlet_bcs_expression is not None:
        logger.info("Applying Dirichlet boundary conditions...")
        dirichlet_bc = dirichlet_bcs_expression(mesh.nodes[mesh.boundary_nodes][:, 0], mesh.nodes[mesh.boundary_nodes][:, 1])
        A, F = apply_dirichlet_bcs(A, F, mesh.boundary_nodes, dirichlet_bc)
    A.tocsr()
    logger.info("Solving the resulting linear system Au = F using conjugate gradient...")
    uh, info = cg(A, F)
    if info == 0:
        logger.info("Conjugate gradient has converged")
    else:
        logger.warning("Convergence not achieved")

    if not display_info:
        logger.setLevel("INFO")

    return uh, A, F