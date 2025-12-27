import numpy as np
from scipy.sparse import lil_matrix

def apply_dirichlet_bcs(A: lil_matrix, F: np.ndarray, dirichlet_nodes: np.ndarray, dirichlet_values: np.ndarray)-> tuple[lil_matrix, np.ndarray]:
    """Modify lhs and rhs to apply dirichlet boundary conditions

    Parameters
    ----------
    A : lil_matrix
        lhs matrix
    F : np.ndarray
        rhs vector
    boundary_nodes : np.ndarray
        boundary nodes
    boundary_values : np.ndarray
        values imposed on boundary nodes

    Returns
    -------
    tuple[lil_matrix, np.ndarray]
        corrected lhs and rhs with bcs applied
    """
    for i in range(len(F)):
        if i not in dirichlet_nodes:
            F[i] -= np.sum(A[i, dirichlet_nodes] * dirichlet_values)

    for node, value in zip(dirichlet_nodes, dirichlet_values):
        A[node, :] = 0
        A[:, node] = 0
        A[node, node] = 1
        F[node] = value
    return A, F