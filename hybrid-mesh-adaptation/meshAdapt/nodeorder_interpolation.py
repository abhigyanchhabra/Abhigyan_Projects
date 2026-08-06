from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from .triangulation import Triangulation

from meshAdapt import Triangulation, searchTriangle, baryCentricCoordinates
from meshAdapt.meshIO import importIn2D, importMesh
import time

def read_bb_values(filename: str) -> List[float]:
    """
    Reads scalar values from a .bb file.
    The first line (header) is skipped, and all subsequent values are parsed as floats.
    """
    values = []
    with open(filename, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header line
        for line in lines:
            values.extend(map(float, line.strip().split()))  # Split and convert each value to float
    return values

def write_bb_values(filename: str, values: List[float]):
    """
    Writes scalar values to a .bb file with a header.
    The format of the header is: "2 1 <num_values> 2"
    Each value is written with 8 decimal precision.
    """
    with open(filename, 'w') as f:
        f.write(f"2 1 {len(values)} 2\n")  # Write the header line
        for val in values:
            f.write(f"{val:.8f}\n")  # Write each value on a new line

def node_order_interpolate(
    geometry_file: str,
    mesh_old_file: str,
    mesh_new_file: str,
    bb_old_file: str,
    bb_output_file: str
):
    print("############## ORDER INTERPOLATION STARTED ##############")
    time_start = time.time()
    """
    Interpolates scalar values defined at nodes of an old mesh to the nodes of a new mesh.
    Uses barycentric interpolation within the triangle of the old mesh containing each new node.
    
    Parameters:
        geometry_file (str): Path to the input .in2d geometry file.
        mesh_old_file (str): Path to the old mesh (.vol) file.
        mesh_new_file (str): Path to the new mesh (.vol) file.
        bb_old_file (str): Path to the old .bb file with scalar values at old mesh nodes.
        bb_output_file (str): Path to the output .bb file for interpolated values.
    """
    # Load geometry and corresponding meshes
    geometry = importIn2D(geometry_file)
    mesh_old = importMesh(mesh_old_file)
    mesh_new = importMesh(mesh_new_file)

    # Create triangulation objects for both old and new meshes
    tri_old = Triangulation(geometry, mesh_old)
    tri_new = Triangulation(geometry, mesh_new)

    # Extract coordinates and triangle definitions from the old triangulation
    coords_old = tri_old.coordinates
    coords_new = tri_new.coordinates
    triangles_old = tri_old.triangles

    # Read scalar values from the old .bb file
    values_old = read_bb_values(bb_old_file)

    interpolated_values = []
    num_new_nodes = len(coords_new) // 2  # Each node has x and y → 2 entries

    # Loop over each node in the new mesh
    for i in range(num_new_nodes):
        # Get (x, y) coordinates of the current new node
        x, y = coords_new[2 * i], coords_new[2 * i + 1]

        # Find the index of the triangle (in the old mesh) that contains this node
        tri_idx = searchTriangle(x, y, tri_old)
        if isinstance(tri_idx, list):
            if not tri_idx:
                interpolated_values.append(0.0)  # Default/fallback value if no containing triangle is found
                continue
            tri_idx = tri_idx[0]  # Take the first triangle if multiple returned

        # Retrieve indices of the three vertices of the found triangle
        p1_idx = triangles_old[3 * (tri_idx // 3)]
        p2_idx = triangles_old[3 * (tri_idx // 3) + 1]
        p3_idx = triangles_old[3 * (tri_idx // 3) + 2]

        # Get coordinates of the three vertices
        p1 = [coords_old[p1_idx], coords_old[p1_idx + 1]]
        p2 = [coords_old[p2_idx], coords_old[p2_idx + 1]]
        p3 = [coords_old[p3_idx], coords_old[p3_idx + 1]]
        point = [x, y]  # New mesh node point

        # Compute barycentric coordinates of the new node w.r.t the triangle in old mesh
        bary = baryCentricCoordinates(p1, p2, p3, point)

        # Interpolate scalar value using barycentric weights
        val_interp = (
            bary[0] * values_old[p1_idx // 2] +
            bary[1] * values_old[p2_idx // 2] +
            bary[2] * values_old[p3_idx // 2]
        )

        interpolated_values.append(val_interp)  # Store interpolated value

    # Write interpolated values to the output .bb file
    write_bb_values(bb_output_file, interpolated_values)

    time_end = time.time()
    print("Time taken for order interpolation:", time_end - time_start, "seconds")
    print("############## ORDER INTERPOLATION END ##############")


        






