import numpy as np
from pathlib import Path

class Mesh:

    def __init__(self, nodes: np.ndarray, elements: np.ndarray, boundary_nodes: np.ndarray, element_type: str):
        self.nodes = np.asarray(nodes, dtype = float)
        self.elements = np.asarray(elements, dtype = int)
        self.boundary_nodes = np.asarray(boundary_nodes, dtype = int)
        self.number_of_nodes = len(self.nodes)
        self.number_of_boundary_nodes = len(self.boundary_nodes)
        self.number_of_elements = len(self.elements)
        self.element_type = element_type
    
    @classmethod
    def unit_square(cls, nx: int, ny: int, element_type = "P1") -> 'Mesh':
        """_summary_

        Parameters
        ----------
        nx : int
            Mesh refinement along x-axis
        ny : int
            Mesh refinement along y-axis
        element_type : str, optional
            finite element type, by default "P1"

        Returns
        -------
        Mesh
            mesh object

        Raises
        ------
        ValueError
            nx and ny must be at least 1
        """
        
        if nx < 1 or ny < 1:
            raise ValueError("nx and ny must be at least 1")
        
        X, Y = np.meshgrid(np.linspace(0.0, 1.0, nx + 1), np.linspace(0.0, 1.0, ny + 1))
        nodes = np.column_stack([X.ravel(), Y.ravel()])

        boundary_nodes = []
        for i, node in enumerate(nodes):
            if node[0] == 0.0 or node[0] == 1.0 or node[1] == 0.0 or node[1] == 1.0:
                boundary_nodes.append(i)
        
        elements = []
        for j in range(ny):
            for i in range(nx):
                # Node indices for the four corners of this quad
                n0 = j * (nx + 1) + i      # Lower-left
                n1 = n0 + 1                 # Lower-right
                n2 = n0 + (nx + 1)         # Upper-left
                n3 = n2 + 1                 # Upper-right
                
                # Two triangles per quadrilateral
                # Triangle 1: lower-left triangle
                elements.append([n0, n1, n2])
                # Triangle 2: upper-right triangle
                elements.append([n1, n3, n2])
        return cls(nodes, np.array(elements, dtype=int), np.array(boundary_nodes, dtype=int), element_type)
    
    def mesh_to_msh(self, filename: str, output_path: Path, version: str = "2.2") -> None:
        """Generate .msh gmsh compatible file

        Parameters
        ----------
        filename : str
            name of the file
        output_path : Path
            where to save the mesh file
        version : str, optional
            gmsh version, by default "2.2"

        """

        if not filename.endswith('.msh'):
            filename += '.msh'
        filename = output_path.joinpath(filename)
        with open(filename, 'w') as f:
            # Write header
            f.write("$MeshFormat\n")
            f.write(f"{version} 0 8\n")  # version, file-type (0=ASCII), data-size
            f.write("$EndMeshFormat\n")
            
            # Write nodes
            f.write("$Nodes\n")
            f.write(f"{self.number_of_nodes}\n")
            for i, (x, y) in enumerate(self.nodes):
                # Format: node-number x y z
                f.write(f"{i+1} {x:.16e} {y:.16e} 0.0\n")
            f.write("$EndNodes\n")
            
            # Write elements
            f.write("$Elements\n")
            f.write(f"{self.number_of_elements}\n")
            elem_id = 1
            # Write triangular elements (element type 2 = 3-node triangle)
            for elem in self.elements:
                # Element type 2 = 3-node triangle
                # Tag 1: physical entity (domain marker, we use 2)
                # Tag 2: elementary entity (geometric entity, we use 2)
                f.write(f"{elem_id} 2 2 2 2 {elem[0]+1} {elem[1]+1} {elem[2]+1}\n")
                elem_id += 1
            
            f.write("$EndElements\n")
        
        print("Mesh written to {}".format(filename))
        print("  Nodes: {}".format(self.number_of_nodes))
        print("  Boundary Nodes:{}".format(self.number_of_boundary_nodes))
        print("  Triangles: {}".format(self.number_of_elements))

        return None

if __name__ == "__main__":
    mesh = Mesh.unit_square(nx = 10, ny = 10)
    mesh.mesh_to_msh(filename = "mesh.msh", output_path=Path(r"C:\projects\py_fem\tests"))
    print("---------------")
