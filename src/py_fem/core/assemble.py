import numpy as np
from typing import Callable
from scipy.sparse import csr_matrix, lil_matrix
from abc import ABC, abstractmethod
from py_fem.core.mesh import Mesh

class Element(ABC):

    @abstractmethod
    def n_nodes(self):
        pass

    @abstractmethod
    def compute_local_stiffness(self, element_coords):
        pass

    @abstractmethod
    def compute_local_forcing(self, element_coords, f):
        pass


class P1Triangle(Element):

    @property
    def n_nodes(self):
        return 3
    
    def compute_local_stiffness(self, element_coords: np.ndarray) -> np.ndarray:
        """Compute local stiffness matrix

        Parameters
        ----------
        element_coords : np.ndarray
            Triangular element vertices coordinates

        Returns
        -------
        np.ndarray
            local stiffness matrix
        """
        x = element_coords[:, 0]
        y = element_coords[:, 1]

        B = np.array([
            [x[1] - x[0], x[2] - x[0]],
            [y[1] - y[0], y[2] - y[0]]
        ])
        
        det_B = np.linalg.det(B)
        
        B_inv = (1 / det_B)* np.array([
            [y[2] - y[0], -x[2] + x[0]],
            [-y[1] + y[0], x[1] - x[0]]
        ])

        # gradient of basis functions
        grad_phi = [np.array([-1, -1]), np.array([1, 0]), np.array([0, 1])]

        # local stiffness matrix
        A_k = np.zeros(shape=(3, 3), dtype=float)

        for i in range(self.n_nodes):
            for j in range(i + 1):
                A_k[i, j] = np.dot(grad_phi[i], np.matmul(np.matmul(B_inv , B_inv.T), grad_phi[j].T)) * det_B / 2
        return A_k + A_k.T - np.diag(np.diag(A_k))
    
    def compute_local_forcing(self, element_coords: np.ndarray, f: Callable[[float, float], float]) -> np.ndarray:
        """Compute local contribute of the forcing term

        Parameters
        ----------
        element_coords : np.ndarray
            Triangular element vertices coordinates
        f : Callable[[float, float], float]
            forcing term as lambda function

        Returns
        -------
        np.ndarray
            Local contribute of the forcing term
        """

        # For linear triangles, use 1-point quadrature at centroid
        centroid = np.mean(element_coords, axis=0)
        x = element_coords[:, 0]
        y = element_coords[:, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))

        # Evaluate source term at centroid
        f_val = f(centroid[0], centroid[1])
        
        # For linear basis functions: integral of each basis function is area/3
        F_elem = (area / 3) * f_val * np.ones(3)
        
        return F_elem


def assemble_global_stiffness(mesh: Mesh) -> lil_matrix:
    """Assemble global stiffness matrix (Laplacian operator)

    Parameters
    ----------
    mesh : Mesh
        Mesh object

    Returns
    -------
    lil_matrix
        global stiffness matrix

    Raises
    ------
    ValueError
        Invalid element type
    """

    element_type = mesh.element_type
    if element_type == "P1":
        finite_element = P1Triangle()
    else:
        raise ValueError("Invalid element type")

    # Initialize global stiffness matrix
    ndofs = mesh.number_of_nodes # number of degrees of freedom
    A_global = lil_matrix((ndofs, ndofs))
    for elem in mesh.elements:
        coords = mesh.nodes[elem]
        A_local = finite_element.compute_local_stiffness(element_coords = coords)

        for i in range(finite_element.n_nodes):
            I = elem[i]
            for j in range(3):
                J = elem[j]
                A_global[I, J] += A_local[i, j]
    return A_global


def assemble_global_forcing(mesh: Mesh, f: Callable[[float, float], float]) -> np.ndarray:
    """Assemble global forcing term

    Parameters
    ----------
    mesh : Mesh
        Mesh object
    f : Callable[[float, float], float]
        forcing as lambda function

    Returns
    -------
    np.ndarray
        forcing vector

    Raises
    ------
    ValueError
        Invalid element type
    """

    element_type = mesh.element_type
    if element_type == "P1":
        finite_element = P1Triangle()
    else:
        raise ValueError("Invalid element type")
    
    # Initialize global rhs vector
    ndofs = mesh.number_of_nodes # number of degrees of freedom
    F_global = np.zeros(ndofs, dtype = float)

    for elem in mesh.elements:
        coords = mesh.nodes[elem]
        F_local = finite_element.compute_local_forcing(element_coords=coords, f=f)
        for i in range(finite_element.n_nodes):
            I = elem[i]
            F_global[I] += F_local[i]
    return F_global
