from typing import TYPE_CHECKING, Dict, List
if TYPE_CHECKING:
    from .triangulation import Triangulation

def importIn2D(in2dPath: str) -> Dict[str, list]:
    """
    Parses a 2D geometry definition file in `.in2d` format and extracts points and boundary segments.

    The function reads and parses structured geometry data. It supports segments defined by either 
    two or three points and handles associated boundary condition flags.

    Parameters:
        in2dPath (str): Path to the `.in2d` file to import.

    Returns:
        dict[str, list]: A dictionary containing:
            - 'points': A flat list of point coordinates [x0, y0, x1, y1, ...].
            - 'segments': A list of segment dictionaries with keys:
                - 'dl', 'dr': Domain labels on the left and right of the segment.
                - 'np': Number of defining points (2 or 3).
                - 'p1', 'p2', ('p3' optional): Indices of defining points (0-based).
                - 'bcFlag': Boundary condition flag.
    """
    lines = []

    # read all lines
    with open(in2dPath, 'r') as file:
        for line in file:
            lines.append(line.strip().split())

    # fill up points defining geometry
    points = []
    for i in range(len(lines)):
        if len(lines[i]) != 0 and lines[i][0] == 'points':
            j = i+1
            while len(lines[j]) != 0:
                points.append(float(lines[j][1]))
                points.append(float(lines[j][2]))

                j+=1
    
    # fill up boundary segments of geometry
    bdySegments = []
    for i in range(len(lines)):
        if len(lines[i]) != 0 and lines[i][0] == 'segments':
            j=i+1
            while len(lines[j]) != 0:
                flag = lines[j][5].split()
                # if curve is given by 2 or 3 points
                if int(lines[j][2]) == 2:
                    segObject = {
                        'dl': int(lines[j][0]),
                        'dr': int(lines[j][1]),
                        'np': int(lines[j][2]),
                        'p1': int(lines[j][3])-1,
                        'p2': int(lines[j][4])-1,
                        'bcFlag': int(list(lines[j][5])[4])
                    }
                elif int(lines[j][2]) == 3:
                    segObject = {
                        'dl': int(lines[j][0]),
                        'dr': int(lines[j][1]),
                        'np': int(lines[j][2]),
                        'p1': int(lines[j][3])-1,
                        'p2': int(lines[j][4])-1,
                        'p3': int(lines[j][5])-1,
                        'bcFlag': int(list(lines[j][6])[4])
                    }
                else:
                    raise RuntimeError(f"Unknown segment type {int(lines[j][2])}.")
                bdySegments.append(segObject)
                j+=1

    # enclose everything in one object
    In2DObj = {
        'points': points,
        'segments': bdySegments
    }
    # print(points, bdySegments)
    return In2DObj

def importMesh(meshPath: str) -> dict[str]:
    """
    Parses a `.vol` mesh file and extracts the mesh structure including points, triangles, and edge segments.
    
    Parameters:
        meshPath (str): Path to the mesh file to import.

    Returns:
        dict[str, list]: A dictionary with the following keys:
            - 'points': A flat list of point coordinates [x0, y0, x1, y1, ...].
            - 'triangles': A flat list of point indices forming triangles in groups of three.
            - 'edgeSegments': A list of dictionaries for each edge segment containing:
                - 'surfid', 'p1', 'p2', 'sf1', 'sf2', 'ednr1', 'dist1', 'ednr2', 'dist2'.
    """
    # read mesh file
    lines = []
    with open(meshPath, 'r') as file:
        for line in file:
            lines.append(line.strip().split())

    points = []
    triangles = []
    edgeSegments = []

    # for all lines
    for i in range(len(lines)):
        # if surfaceelements
        if len(lines[i]) != 0 and lines[i][0] == 'surfaceelements':
            numElements = int(lines[i+1][0])
            for j in range(i+2, i+2+numElements):
                a = (int(lines[j][5])-1)*2
                b = (int(lines[j][6])-1)*2
                c = (int(lines[j][7])-1)*2
                triangles.append(a)
                triangles.append(b)
                triangles.append(c)

        # if points
        if len(lines[i]) != 0 and lines[i][0] == 'points':
            numPoints = int(lines[i+1][0])
            for j in range(i+2, i+2+numPoints):
                points.append(float(lines[j][0]))
                points.append(float(lines[j][1]))

    # for all lines
    for i in range(len(lines)):
        # if edgesegmentsgi2
        if len(lines[i]) != 0 and lines[i][0] == 'edgesegmentsgi2':
            numEdgeSegs = int(lines[i+1][0])
            for j in range(i+2, i+2+numEdgeSegs):
                edgeSeg = {
                    'surfid': int(lines[j][0]),
                    'p1': int(lines[j][2])-1,
                    'p2': int(lines[j][3])-1,
                    'sf1': int(lines[j][6]),
                    'sf2': int(lines[j][7]),
                    'ednr1': int(lines[j][8])-1,
                    'dist1': float(lines[j][9]),
                    'ednr2': int(lines[j][10])-1,
                    'dist2': float(lines[j][11])
                }
                edgeSegments.append(edgeSeg)

    # enclose everything in one object
    meshObj = {
        'points': points,
        'triangles': triangles,
        'edgeSegments': edgeSegments
    }
    # print(points, "####################", triangles, "################", edgeSegments)
    return meshObj

def importMetric(metricPath: str) -> List[float]:
    """
    Parses a metric tensor file and returns the metric values as a flat list.

    The function assumes the file contains a header line followed by rows of 
    three floating-point values (representing components of 2D metric tensors).

    Parameters:
        metricPath (str): Path to the metric file to import.

    Returns:
        List[float]: A flat list of metric tensor components [m11, m12, m22, ...].
    """
    metric = []
    if metricPath:
        with open(metricPath, 'r') as file:
            for i, line in enumerate(file):
                # skipping header line
                if i > 0:
                    a, b, c = line.strip().split()
                    metric.append(float(a))
                    metric.append(float(b))
                    metric.append(float(c))
    # print(metric)
    return metric

def exportMesh(triangulation: 'Triangulation', fileName: str, fileFormat: str) -> None:
    """
    Exports a 2D triangulation mesh to a file in the specified format (currently supports only 'vol').

    Parameters:
        triangulation (Triangulation): A triangulation object with the following attributes:
            - triangles (list[int]): Flat list of vertex indices forming triangles.
            - coordinates (list[float]): Flat list of vertex coordinates [x0, y0, x1, y1, ...].
            - boundary (list[dict]): List of edge segment dictionaries with mesh boundary data.
        fileName (str): The base name of the file to write (without extension).
        fileFormat (str): The file format to write (only 'vol' is currently supported).

    Returns:
        None

    Raises:
        ValueError: If an unsupported file format is provided.
    """
    triangles = triangulation.triangles
    coords = triangulation.coordinates
    
    file_strings = []

    if fileFormat == 'vol':
        file_strings.append("mesh3d\ndimension\n2\ngeomtype\n0\n")
        file_strings.append('# surfnr    bcnr    domin    domout    np      p1      p2      p3')
        file_strings.append('surfaceelements')
        file_strings.append(str(len(triangles) // 3))
        
        for i in range(0, len(triangles), 3):
            file_strings.append(f"{2:>8}{1:>8}{0:>8}{0:>8}{3:>8}{(triangles[i] // 2 + 1):>8}{(triangles[i+1] // 2 + 1):>8}{(triangles[i+2] // 2 + 1):>8}")

        file_strings.append('\n# matnr    np    p1    p2    p3    p4')
        file_strings.append('volumeelements')
        file_strings.append('0\n')

        file_strings.append('# surfid       0      p1       p2   trinum1  trinum2 domin/sf1 domout/sf2 ednr1    dist1     ednr2     dist2')
        file_strings.append('edgesegmentsgi2')

        edgeSegments = triangulation.boundary
        numSeg = len(edgeSegments)
        
        file_strings.append(str(numSeg))

        for i in range(numSeg):
            surfId = edgeSegments[i]['surfid']
            p1 = edgeSegments[i]['p1']+1
            p2 = edgeSegments[i]['p2']+1
            sf1 = edgeSegments[i]['sf1']
            sf2 = edgeSegments[i]['sf2']
            ednr1 = edgeSegments[i]['ednr1']+1
            ednr2 = edgeSegments[i]['ednr2']+1
            dist1 = edgeSegments[i]['dist1']
            dist2 = edgeSegments[i]['dist2']
            file_strings.append(f"{surfId:>8}{0:>8}{p1:>8}{p2:>8}{-1:>9}{-1:>9}{sf1:>9}{sf2:>9}{ednr1:>8}{dist1:>24.16e}{ednr2:>8}{dist2:>24.16e}")

        file_strings.append('#     X     Y     Z')
        file_strings.append('points')
        file_strings.append(str(len(coords) // 2))
        
        for i in range(0, len(coords), 2):
            file_strings.append(f"{coords[i]:>24.16e}{coords[i+1]:>29.16e}{0:>29.16e}")
        
        file_strings.append('materials\n1\n1 domain1\n\nendmesh')

        # Save the file
        with open(fileName + "." + fileFormat, 'w') as file:
            file.write("\n".join(file_strings))

    elif fileFormat == 'su2':
        # reference: https://su2code.github.io/docs/Mesh-File/
        file_strings.append("NDIME= 2\n")

        # writing points
        file_strings.append(f'NPOIN= {str(len(coords) // 2)}')
        for i in range(0, len(coords), 2):
            file_strings.append(f"{coords[i]:>24.16e}{coords[i+1]:>29.16e}")
        
        # writing elements
        file_strings.append(f'NELEM= {str(len(triangles) // 3)}')
        for i in range(0, len(triangles), 3):
            file_strings.append(f"{5:>8}{(triangles[i] // 2):>8}{(triangles[i+1] // 2):>8}{(triangles[i+2] // 2):>8}")

        # collecting markers
        edgeSegments = triangulation.boundary
        markers = {}
        nMarks = 0
        for edge in edgeSegments:
            surfId = edge['surfid']

            # create new list for this surfId if doesnt exist yet
            if surfId not in markers:
                nMarks += 1
                markers[surfId] = []
            
            markers[surfId].append(edge)

        # writing markers
        file_strings.append(f'NMARK= {str(nMarks)}')
        for surfid in markers:
            file_strings.append(f'MARKER_TAG= {surfid}')
            file_strings.append(f'MARKER_ELEMS= {len(markers[surfid])}')
            for element in markers[surfid]:
                p1 = element['p1']
                p2 = element['p2']
                file_strings.append(f'{3:>8}{p1:>8}{p2:>8}')

        # Save the file
        with open(fileName + "." + fileFormat, 'w') as file:
            file.write("\n".join(file_strings))

    elif fileFormat == 'mesh':      # Warning: mesh format reference not found,
                                    # format just understood from refine mesh as example
        file_strings.append("MeshVersionFormatted 2\nDimension 2\n")

        # writing points
        file_strings.append("Vertices")
        file_strings.append(str(len(coords) // 2))
        for i in range(0, len(coords), 2):
            file_strings.append(f"{coords[i]:>24.16e}{coords[i+1]:>29.16e}{1:>8}")
        
        # writing elements
        file_strings.append("Triangles")
        file_strings.append(str(len(triangles) // 3))
        for i in range(0, len(triangles), 3):
            file_strings.append(f"{(triangles[i] // 2 + 1):>8}{(triangles[i+1] // 2 + 1):>8}{(triangles[i+2] // 2 + 1):>8}{0:>8}")

        # collecting edges
        edgeSegments = triangulation.boundary
        edges = {}
        for edge in edgeSegments:
            surfId = edge['surfid']

            # create new list for this surfId if doesnt exist yet
            if surfId not in edges:
                edges[surfId] = []
            
            edges[surfId].append(edge)

        # writing edges
        nEdges = len(edgeSegments)
        file_strings.append("Edges")
        file_strings.append(str(nEdges))
        for surfid in edges:
            for element in edges[surfid]:
                p1 = element['p1']+1
                p2 = element['p2']+1
                file_strings.append(f'{p1:>8}{p2:>8}{surfid:>8}')

        # Save the file
        with open(fileName + "." + fileFormat, 'w') as file:
            file.write("\n".join(file_strings))
    
    else:
        raise ValueError(f'Given file format {fileFormat} is not accepted.')
    
def exportMetric(metricMesh, fileName):

    fileStrings = []

    fileStrings.append(f"{len(metricMesh)//3} 3")

    for i in range(0, len(metricMesh), 3):
        fileStrings.append(f"{metricMesh[i]:>24.16e}{metricMesh[i+1]:>24.16e}{metricMesh[i+2]:>24.16e}")

    # Save the file
    with open(fileName, 'w') as file:
        file.write("\n".join(fileStrings))