import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from py_fem.core.mesh import Mesh


def plot_solution_2d(mesh: Mesh, solution: np.ndarray, title: str = 'FEM Solution', cmap: str ='viridis', 
                     show_mesh: bool = False, levels: int = 40, ax: plt.Axes = None) -> None:
    """
    Plot 2D solution as a filled contour plot.
    
    Parameters:
    -----------
    mesh : Mesh object
        Must have: nodes (n_nodes, 2), elements (n_elements, n_nodes_per_elem)
    solution : ndarray, shape (n_nodes,)
        Solution vector at nodes
    title : str
        Plot title
    cmap : str
        Colormap name
    show_mesh : bool
        Whether to overlay mesh edges
    levels : int
        Number of contour levels
    ax: plt.Axes
        matplotlib.pyplot axes object
    """
    # Create triangulation
    # For P2 elements, use only the vertex nodes (first 3 of each element)
    if mesh.elements.shape[1] == 6:  # P2 triangles
        tri_elements = mesh.elements[:, :3]
        triang = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], tri_elements)
        # For P2, we should interpolate, but for simplicity plot vertex values
        z = solution
    elif mesh.elements.shape[1] == 3:  # P1 triangles
        triang = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
        z = solution
    else:
        raise ValueError("Only triangular elements supported for this plot type")
    
    # Filled contour plot
    contourf = ax.tricontourf(triang, z, levels=levels, cmap=cmap)
    
    # Optional: show mesh edges
    if show_mesh:
        ax.triplot(triang, 'k-', linewidth=0.3, alpha=0.4)
    
    # Colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Solution value', rotation=270, labelpad=20)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    return