from py_fem.core.mesh import Mesh
import numpy as np
def compute_L2_error(mesh: Mesh, numerical_sol: np.ndarray, analytical_sol: np.ndarray) -> float:
    """Compute the L2 norm of the difference between analytical and numerical solution

    Parameters
    ----------
    mesh : Mesh
        mesh object
    numerical_sol : np.ndarray
        numerical solution at mesh nodes
    analytical_sol : np.ndarray
        analytical solution evaluated at mesh nodes

    Returns
    -------
    float
        L2 norm of the error i.e. ||u - u_h||_L2

    Raises
    ------
    ValueError
        Invalid shape of analytical and numerical solutions
    """
    
    if numerical_sol.shape != analytical_sol.shape:
        raise ValueError("analytical and numerical solution must have the same dimension")
    
    error_squared = 0.0

    for elem in mesh.elements:
        elem_coords = mesh.nodes[elem]
        u_h_elem = numerical_sol[elem]
        u_exact_elem = analytical_sol[elem]
        
        # Compute element contribution
        error_squared += compute_element_L2_error_squared(
            elem_coords, u_h_elem, u_exact_elem
        )
    
    L2_error = np.sqrt(error_squared)
    return L2_error


def compute_element_L2_error_squared(elem_coords: np.ndarray, u_h_elem: np.ndarray, u_exact_elem: np.ndarray):
    """
    Compute element contribution to L2 error squared for P1 triangle.
    
    For P1 elements, both u_h and u_exact are represented as:
    u(x,y) = u0*φ0 + u1*φ1 + u2*φ2
    
    The integral ∫∫_T (u_h - u_exact)² dA can be computed exactly.
    
    Parameters:
    -----------
    elem_coords : ndarray, shape (3, 2)
        Element node coordinates
    u_h_elem : ndarray, shape (3,)
        Numerical solution at element nodes
    u_exact_elem : ndarray, shape (3,)
        Analytical solution at element nodes
        
    Returns:
    --------
    error_sq : float
        Element contribution to L2 error squared
    """
    x = elem_coords[:, 0]
    y = elem_coords[:, 1]
    
    # Element area
    area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - 
                     (x[2] - x[0]) * (y[1] - y[0]))
    
    # Error at nodes
    e = u_h_elem - u_exact_elem  # e = [e0, e1, e2]
    
    # For P1 elements: ∫∫_T φi*φj dA = area * M_ij
    # where M is the element mass matrix
    M = np.array([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ]) / 12.0
    
    # ∫∫_T (error)² dA = e^T * (area * M) * e
    error_sq = area * np.dot(e, M @ e)
    
    return error_sq