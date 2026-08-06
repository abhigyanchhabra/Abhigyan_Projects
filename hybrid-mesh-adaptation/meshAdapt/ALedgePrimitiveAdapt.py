from typing import Optional
import numpy as np
import time
from meshAdapt import Triangulation, aflrAdapt, exportMesh, blMesh, searchTriangle, \
      triangleSubdivide, bdyTriangleSubdivide, isOnSegment, edgePrimitiveAdapt

def triangle_area(mesh, triStart):
    """Compute area of triangle starting at triStart index."""
    n1 = mesh.triangles[triStart]
    n2 = mesh.triangles[triStart + 1]
    n3 = mesh.triangles[triStart + 2]

    x1, y1 = mesh.coordinates[n1], mesh.coordinates[n1 + 1]
    x2, y2 = mesh.coordinates[n2], mesh.coordinates[n2 + 1]
    x3, y3 = mesh.coordinates[n3], mesh.coordinates[n3 + 1]

    return 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))

def is_bl_triangle(mesh, triStart):
    """Check if triangle belongs to boundary layer region."""
    if not mesh.blFlags:
        return False
    return mesh.blFlags[triStart // 3] == 0

time_start = time.time()

def ALedgePrimitiveAdapt(
        in2dPath: str, 
        bgMesh: Triangulation, 
        bcflag: int, 
        nBLLayers: int, 
        splitMergePasses: int, 
        r: Optional[float]=None,
        delta0: Optional[float]=None) -> Triangulation:
    """
    Generates a BL mesh composed of right triangles and then switches to edge primitive adaptation

    Parameters:
        in2dPath (str): Path to the .in2d file containing the geometry and boundary conditions.
        bgMesh (Triangulation): A triangulation object containing coordinates, triangles, metric field, and other mesh-related data.
        bcFlag (int): The boundary condition flag indicating viscous boundary segments.
        nBLLayers (int): Number of layers in the boundary layer.
        splitMergePasses (int): Number of passes for the edge primitive adaptation phase.
        r (float): Growth rate for successive boundary layers.
        delta0 (float): Initial boundary layer thickness.
        
    Returns:
        Triangulation: A triangulation object representing the adapted mesh after edge primitive adaptation.
    """

    # al+aflr
    bdyMesh = aflrAdapt(in2dPath, bgMesh.metric, iterations=0, runParallel=True)
    flags, newMetric = blMesh(bdyMesh, bcFlag=bcflag, nLayers=nBLLayers, bgMesh=bgMesh, r=r, delta0=delta0)
    bdyMesh.blFlags = flags
    exportMesh(bdyMesh, 'netgen_bl', 'vol')

    bgMesh_points = []
    bgMesh_metric = []

    # build points
    for i in range(0, len(bgMesh.coordinates)-1, 2):
        bgMesh_points.append([bgMesh.coordinates[i], bgMesh.coordinates[i+1]])

    # build metrics
    for i in range(0, len(bgMesh.metricMesh)-2, 3):
        bgMesh_metric.append([bgMesh.metricMesh[i], bgMesh.metricMesh[i+1], bgMesh.metricMesh[i+2]])

    assert len(bgMesh_points) == len(bgMesh_metric)

    print("Populating BL mesh with background mesh points...")
    for i, point in enumerate(bgMesh_points):

        # if i == 0: break

        # Skip if point is too close to existing node
        coords = np.array(bdyMesh.coordinates).reshape(-1, 2)
        dist = np.hypot(coords[:, 0] - point[0], coords[:, 1] - point[1])
        if np.min(dist) < 1e-8: continue

        # Locate containing triangle or edge
        triangleIndex = searchTriangle(point[0], point[1], triangulation=bdyMesh, startTriIdx=None)
        if triangleIndex == -1: continue

        # Zero-area protection
        if isinstance(triangleIndex, int):
            if triangle_area(bdyMesh, triangleIndex) < 1e-10: continue

        elif isinstance(triangleIndex, list):
            triA, triB = triangleIndex
            if (triangle_area(bdyMesh, triA) < 1e-10 or triangle_area(bdyMesh, triB) < 1e-10): continue

        # Boundary layer protection
        if isinstance(triangleIndex, int):
            if is_bl_triangle(bdyMesh, triangleIndex): continue

        elif isinstance(triangleIndex, list):
            triA, triB = triangleIndex
            if is_bl_triangle(bdyMesh, triA) or is_bl_triangle(bdyMesh, triB): continue

        proposedMetric = bgMesh_metric[i]

        # Subdivision logic
        # Case A: Point inside triangle or on boundary edge
        if isinstance(triangleIndex, int):
            triStart = triangleIndex
            boundaryEdgeLocalIdx = None

            # Check if point lies on a boundary edge of this triangle
            for k in range(3):
                if bdyMesh.adjacents[triStart + k] == -1:
                    idx1 = bdyMesh.triangles[triStart + k]
                    idx2 = bdyMesh.triangles[triStart + (k + 1) % 3]
                    x1, y1 = bdyMesh.coordinates[idx1], bdyMesh.coordinates[idx1 + 1]
                    x2, y2 = bdyMesh.coordinates[idx2], bdyMesh.coordinates[idx2 + 1]
                    if isOnSegment(x1, y1, point[0], point[1], x2, y2):
                        boundaryEdgeLocalIdx = k
                        break

            # Boundary edge case
            if boundaryEdgeLocalIdx is not None:
                uIdx = triStart + boundaryEdgeLocalIdx
                bdyTriangleSubdivide(bdyMesh, point, uIdx, proposedMetric)

            # Point inside triangle case
            else:
                triangleSubdivide(bdyMesh, point, triStart, proposedMetric)

        # Case B: Point lies on interior edge
        elif isinstance(triangleIndex, list):
            triA, triB = triangleIndex
            triangleSubdivide(bdyMesh, point, [triA, triB], proposedMetric)
    exportMesh(bdyMesh, 'netgen_bl_bgPoints', 'vol')

    print("Performing edge-primitive adaptation...")
    bl_sm = edgePrimitiveAdapt(bdyMesh, splitMergePasses)
    # exportMesh(bl_sm, 'netgen_bl_sm', 'vol')

    time_end = time.time()
    print(f"Total time: {time_end - time_start:.2f} seconds")

    return bl_sm