import numpy as np
from scipy.spatial import Delaunay
import time
from typing import Dict, Optional, Callable, List

from .triangulationSearch import adjacencyList, boundaryProject, closestBoundaryPoint, searchTriangle, collectEdges, findNearestTriangle
from .metric import logMetric, expMetric, baryCentricCoordinates, discretizeSegment  # edit by Abhigyan: added discretizeSegment for boundary discretization methods
from .meshIO import importIn2D, importMesh, importMetric
from .delaunay import Delaunator, constrainedDelaunay
from .pointQuadTree import buildPointQuadTree
from .elementADT import buildElementADT

class Triangulation:
    def __init__(self, in2dObj: Dict, meshObj: Dict) -> None:
        """
        A class to manage and operate on a 2D triangular mesh, including preprocessing steps such as 
        boundary point extraction, triangle attachment, adjacency list construction, and spatial indexing 
        via a quadtree.

        Attributes:
            in2dObj (dict): The input object containing 2D mesh-related data.
            meshObj (dict): Dictionary with mesh data including points, triangles and boundary edges.
            coordinates (List[float]): Flattened list of x and y coordinates for mesh points.
            triangles (List[int]): Flattened list of triangles, each defined by three point indices.
            boundary (List[dict]): List of boundary segments with 'p1' and 'p2' point indices.
            bdyPoints (List[int]): List of boundary point indices (doubled).
            adjPointTris (List[int]): List mapping each point to an adjacent triangle.
            epsilon (float): Tolerance for determining proximity in boundary triangle localization.
            metricMesh (Optional[Any]): A background mesh for defining a spatial metric.
            metricLog (Optional[Any]): Logarithmic metric.
            bgMetricFunction (Optional[Callable]): Function defining the background metric.
            adjacents (List[int]): Adjacency list mapping each triangle edge to index of adjacent triangle in triangles list.
            xMin (float): Minimum x-coordinate of the mesh.
            xMax (float): Maximum x-coordinate of the mesh.
            yMin (float): Minimum y-coordinate of the mesh.
            yMax (float): Maximum y-coordinate of the mesh.
            _prevTriIdx (int): Previously used triangle index for localization optimization.
            flags (Optional[Any]): Optional data structure for tagging or marking mesh triangles.
            quadTree (PointQuadTree): Spatial index (quadtree) built for efficient point location.
            elementADT (ElementADT): Spatial index (alternating digital tree) built for efficient element location.
            edges (List[int]): List of mesh edges represented by index in triangles list.
        """
        # store input objects
        self.in2dObj = in2dObj
        self.meshObj = meshObj
        
        # extract mesh data
        self.coordinates = meshObj['points']
        self.triangles = meshObj['triangles']
        self.boundary = meshObj['edgeSegments']

        # build boundary points (preprocessing)
        self.bdyPoints = []
        in2dSegs = self.boundary
        for i in range(len(in2dSegs)):
            self.bdyPoints.append(in2dSegs[i]['p1']*2)
            self.bdyPoints.append(in2dSegs[i]['p2']*2)

        # attach triangles to boundary points and check for errors
        self.adjPointTris = self.attachTriangles()
        if -1 in self.adjPointTris: raise RuntimeError

        # epsilon for nearby triangles (preprocessing)
        self.epsilon = self.calculateEpsilon()

        # metric defined by a background mesh
        self.metricMesh = None
        self.metricLog = None

        # background metric function 
        self.bgMetricFunction = None

        # build adjacents
        self.adjacents = adjacencyList(self.triangles, len(self.coordinates)//2)

        # plotting related constants
        self.xMin = min(self.coordinates[::2])
        self.xMax = max(self.coordinates[::2])
        self.yMin = min(self.coordinates[1::2])
        self.yMax = max(self.coordinates[1::2])

        # triangle index previously used for triangle localization
        self._prevTriIdx = 0

        self.flags = None
        # bl triangle flags
        self.blFlags = None
        # edit by Abhigyan
        self.protectedSplitEdges = None  # set of frozenset({p1, p2}) coord-array indices for edges splitPass must never split; None means no protection

        # build point quadtree.
        self.quadTree = buildPointQuadTree(self)

        # build element ADT (alternating digital tree).
        self.elementADT = buildElementADT(self)

        # build edge list
        self.edges = collectEdges(self)

    @classmethod
    def fromIn2D(
        cls, 
        in2dPath: str,
        metricFunction: Optional[Callable[[float, float], List[float]]]=None) -> 'Triangulation':
        """
        Creates a Triangulation object from an In2D file path.

        This method handles loading the In2D input, generating a default mesh,
        constructing the triangulation, enforcing constrained Delaunay conditions,
        and building the spatial quadtree index.

        Parameters:
            in2dPath (str): Path to the In2D input file.
            metricFunction (Optional[Callable[[float, float], List[float]]]): A callable metric function to assign node metrics.

        Returns:
            Triangulation: An initialized Triangulation object with preprocessing completed.
    
        """
        in2dObj = importIn2D(in2dPath)
        meshObj = cls._defaultMesh(in2dObj)

        # build triangulation
        tri = cls(in2dObj, meshObj)
        
        # build adjacents
        tri.adjacents = adjacencyList(tri.triangles, len(tri.coordinates)//2)

        # constrained delaunay
        constrainedDelaunay(tri)

        # build point quadtree.
        tri.quadTree = buildPointQuadTree(tri)

        # build element ADT (alternating digital tree).
        tri.elementADT = buildElementADT(tri)

        # if background metric funtion is available
        if metricFunction:
            tri.bgMetricFunction = metricFunction

            tri.assignNodeMetric()

        return tri

    @classmethod
    def fromMesh(
        cls, 
        in2dPath: str, 
        meshPath: str, 
        metricPath: Optional[str]=None, 
        metricFunction: Optional[Callable[[float, float], List[float]]]=None) -> 'Triangulation':
        """
        Creates a Triangulation object from specified mesh and optional metric data.

        Loads the In2D input and mesh file, constructs the triangulation, and optionally
        processes a background metric either from a file or a provided function.

        Parameters:
            in2dPath (str): Path to the In2D input file.
            meshPath (str): Path to the mesh file.
            metricPath (Optional[str]): Path to the metric file (if provided).
            metricFunction (Optional[Callable[[float, float], List[float]]]): A callable metric function to assign node metrics.

        Returns:
            Triangulation: An initialized Triangulation object with optional metric information.
        """
        in2dObj = importIn2D(in2dPath)
        meshObj = importMesh(meshPath)

        # build triangulation
        tri = cls(in2dObj, meshObj)

        # if metric available
        if metricPath:
            # read metric file
            tri.metricMesh = importMetric(metricPath)
            # calculate metric log 
            tri.metricLog = tri.assignNodeMetricLog()
        # if background metric funtion is available
        elif metricFunction:
            tri.bgMetricFunction = metricFunction

            tri.assignNodeMetric()

        return tri

    def metric(self, x: float, y: float) -> List[float]:
        """
        Computes the metric tensor at a given (x, y) point in the mesh.

        If a background metric function is provided, it will be used directly.
        Otherwise, the method performs triangle localization, computes barycentric 
        coordinates, and uses Log-Euclidean interpolation of the logarithmic metric 
        tensors at the triangle vertices to evaluate the metric at the given point.

        Args:
            x (float): X-coordinate of the query point.
            y (float): Y-coordinate of the query point.

        Returns:
            List[float]: The interpolated metric tensor at the point in the form 
                        [m11, m12, m22], representing the symmetric 2D metric matrix:
                        
                            | m11  m12 |
                            | m12  m22 |
        
        Raises:
            ValueError: If the point (x, y) is not inside any triangle in the background mesh.
        """
        # if metric function exists on this triangulation, use that
        if self.bgMetricFunction:
            return self.bgMetricFunction(x, y)

        initialTriangleIdx = self._prevTriIdx
        # find containing triangle in the background mesh
        triangleIdx = searchTriangle(x, y, self, initialTriangleIdx)
        if triangleIdx == -1:
            # Point lies just outside the background mesh (e.g. a new point
            # projected onto a curved boundary that bulges past the
            # background mesh's piecewise-linear edge). Fall back to the
            # nearest triangle and extrapolate the metric from it.
            triangleIdx = findNearestTriangle(x, y, self)

        # if point is on edge shared by two triangles, take point inside first one
        if type(triangleIdx) == list: 
            triangleIdx = triangleIdx[0]
        self._prevTriIdx = triangleIdx
        
        # find coordinates of three points
        point0 = [self.coordinates[self.triangles[triangleIdx]], self.coordinates[self.triangles[triangleIdx]+1]]
        point1 = [self.coordinates[self.triangles[triangleIdx+1]], self.coordinates[self.triangles[triangleIdx+1]+1]]
        point2 = [self.coordinates[self.triangles[triangleIdx+2]], self.coordinates[self.triangles[triangleIdx+2]+1]]

        # find barycentric coordinates
        baryCoords = baryCentricCoordinates(point0, point1, point2, [x, y])
        
        # Log Eucledian interp
        # find log metric at three vertices
        lm0 = [self.metricLog[int(self.triangles[triangleIdx]*1.5)], self.metricLog[int(self.triangles[triangleIdx]*1.5)+1], self.metricLog[int(self.triangles[triangleIdx]*1.5)+2]]
        lm1 = [self.metricLog[int(self.triangles[triangleIdx+1]*1.5)], self.metricLog[int(self.triangles[triangleIdx+1]*1.5)+1], self.metricLog[int(self.triangles[triangleIdx+1]*1.5)+2]]
        lm2 = [self.metricLog[int(self.triangles[triangleIdx+2]*1.5)], self.metricLog[int(self.triangles[triangleIdx+2]*1.5)+1], self.metricLog[int(self.triangles[triangleIdx+2]*1.5)+2]]

        # finding sum with barycentric coordinates
        mSum = [baryCoords[0]*lm0[0] + baryCoords[1]*lm1[0] + baryCoords[2]*lm2[0],
                baryCoords[0]*lm0[1] + baryCoords[1]*lm1[1] + baryCoords[2]*lm2[1],
                baryCoords[0]*lm0[2] + baryCoords[1]*lm1[2] + baryCoords[2]*lm2[2]]

        # finding metric exponential
        mX = expMetric(mSum)

        # linear interp
        # metric0 = [self.metricMesh[int(self.triangles[triangleIdx]*1.5)], self.metricMesh[int(self.triangles[triangleIdx]*1.5)+1], self.metricMesh[int(self.triangles[triangleIdx]*1.5)+2]]
        # metric1 = [self.metricMesh[int(self.triangles[triangleIdx+1]*1.5)], self.metricMesh[int(self.triangles[triangleIdx+1]*1.5)+1], self.metricMesh[int(self.triangles[triangleIdx+1]*1.5)+2]]
        # metric2 = [self.metricMesh[int(self.triangles[triangleIdx+2]*1.5)], self.metricMesh[int(self.triangles[triangleIdx+2]*1.5)+1], self.metricMesh[int(self.triangles[triangleIdx+2]*1.5)+2]]
        
        # m1 = baryCoords[0]*metric0[0] + baryCoords[1]*metric1[0] + baryCoords[2]*metric2[0]
        # m2 = baryCoords[0]*metric0[1] + baryCoords[1]*metric1[1] + baryCoords[2]*metric2[1]
        # m3 = baryCoords[0]*metric0[2] + baryCoords[1]*metric1[2] + baryCoords[2]*metric2[2]
        # mX = [m1, m2, m3]

        return mX
        
    def assignNodeMetricLog(self) -> List[float]:
        """
        Computes the logarithmic representation of the metric tensor at each mesh node metric.

        Returns:
            List[float]: A flattened list containing the logarithmic metric tensors 
                        for all nodes in the mesh, stored in triplets [m11, m12, m22].
        """
        metricMesh = self.metricMesh
        metricLog = []
        # precalculating metric log
        for i in range(0, len(metricMesh), 3):
            mLn = logMetric([metricMesh[i], metricMesh[i+1], metricMesh[i+2]])
            metricLog += mLn
        return metricLog

    def assignNodeMetric(self) -> None:
        """
        Evaluates the metric tensor at each node in the mesh using the background metric function.

        This method iterates over all node coordinates, applies the `metric()` method
        to compute the metric tensor, and stores the result in `self.metricMesh`.
        It also computes and stores the logarithmic form in `self.metricLog`.

        Returns:
            None
        """
        xs = self.coordinates[::2]
        ys = self.coordinates[1::2]
        metricArray = []
        for x, y in zip(xs, ys):    
            metric = self.metric(x, y)
            metricArray += metric

        self.metricMesh = metricArray
        self.metricLog = self.assignNodeMetricLog()

    # def delaunay(self):
    #     # build delaunay triangulation
    #     xs = self.coordinates[::2]
    #     ys = self.coordinates[1::2]

    #     coords = []
    #     for idx in range(len(xs)):
    #         coords.append([xs[idx], ys[idx]])
    #     coords = np.array(coords)

    #     # SciPy Delaunay triangulator
    #     delTri = Delaunay(coords)
    #     self.triangles = [i*2 for row in delTri.simplices for i in row]

    #     # delaunator in developmenmt
    #     # delTri = Delaunator(coords)
    #     # triangulation.triangles = [x*2 for x in delTri.triangles]
        
    #     # Delaunay of scipy gives flat triangles sometimes so they are removed here
    #     from .aflr import checkFlatTriangles
    #     # collecting all flat triangles
    #     flatTriIdx = checkFlatTriangles(self.coordinates, self.triangles)
    #     # saving only valid triangles
    #     if len(flatTriIdx) > 0:
    #         print("Warning: Flat triangles formed, cleaning...")
    #         validTris = []
    #         for idx in range(0, len(self.triangles), 3):
    #             if idx not in flatTriIdx:
    #                 validTris.append(self.triangles[idx])
    #                 validTris.append(self.triangles[idx+1])
    #                 validTris.append(self.triangles[idx+2])
    #         self.triangles = validTris
        
    #     # adding valid adjacency list
    #     self.adjacents = adjacencyList(self.triangles, len(self.coordinates)//2)
    #     # build attached triangles
    #     self.adjPointTris = self.attachTriangles()
    #     constrainedDelaunay(self)
    
    def attachTriangles(self) -> List[int]:
        """
        Associates each mesh point with one adjacent triangle.

        Parameters:
            None

        Returns:
            List[int]: A list where each index corresponds to a point in the mesh and the value
                    is the index of an adjacent triangle.
        """
        adjPointTris = [-1]*(len(self.coordinates)//2)
        for triIdx in range(0, len(self.triangles), 3):
            # for all three points
            for i in range(triIdx, triIdx+3):
                pointIdx = self.triangles[i]
                if adjPointTris[pointIdx//2] == -1:
                    adjPointTris[pointIdx//2] = triIdx
        return adjPointTris
    
    def calculateEpsilon(self) -> float:
        """
        Calculates a small epsilon value based on the minimum perpendicular distance
        from control points to line segments, used for geometric tolerance.

        This value is determined by examining curved segments (those with a midpoint 'p2')
        in the input geometry, and measuring how far the control point deviates from the
        straight line formed by the endpoints.

        Parameters:
            None

        Returns:
            float: A small positive value representing the minimum allowed geometric tolerance.
        """
        coords = self.in2dObj['points']
        dMin = 1e-10
        for seg in self.in2dObj['segments']:
            if 'p3' in seg:
                p1x = coords[seg['p1']*2]
                p1y = coords[seg['p1']*2+1]
                pcx = coords[seg['p2']*2]
                pcy = coords[seg['p2']*2+1]
                p2x = coords[seg['p3']*2]
                p2y = coords[seg['p3']*2+1]
                
                d = abs((p2y-p1y)*pcx - (p2x-p1x)*pcy + p2x*p1y - p2y*p1x)/((p2x-p1x)**2 + (p2y-p1y)**2)**0.5
                d /=2

                if d > dMin:
                    dMin = d
        return dMin

    # edit by Abhigyan: methods below move functionality from triangulate_quad_geometry.py into the class

    def discretize_boundary_except(self, protected_bc: int) -> None:
        """Discretizes all boundary segments except those tagged with protected_bc.

        Leaves every segment whose geometry bcFlag equals protected_bc exactly as
        given (no Steiner points), so the quad layers' outer boundary keeps the
        exact node count it was built with.  All other segments are subdivided
        by discretizeSegment as usual.
        """
        segs = len(self.meshObj['edgeSegments'])  # original segment count, captured before subdivision grows the list
        currentEdge = 0  # index into the (growing) edgeSegments list of the next original segment to process
        for _ in range(segs):
            geoSegIdx = self.meshObj['edgeSegments'][currentEdge]['ednr1']  # index into in2dObj['segments'] for this mesh edge
            if self.in2dObj['segments'][geoSegIdx]['bcFlag'] == protected_bc:
                currentEdge += 1  # skip: leave this bc=protected_bc segment unsplit
                continue
            currentEdge += discretizeSegment(self, currentEdge)  # advance by however many sub-segments this one became

    def mark_protected_split_edges(self, protected_bc: int) -> set:
        """Populates self.protectedSplitEdges with all edges belonging to protected_bc segments.

        Sets self.protectedSplitEdges to a set of frozenset({p1, p2}) coordinate-array
        index pairs (matching triangulation.triangles units) that splitPass must never split.
        Returns the same set for convenience.
        """
        protected_edges = set()  # frozenset({p1, p2}) coord-array indices for every bc=protected_bc mesh edge
        for edgeSeg in self.meshObj['edgeSegments']:
            geoSeg = self.in2dObj['segments'][edgeSeg['ednr1']]  # geometry segment this mesh edge came from, for its bcFlag
            if geoSeg['bcFlag'] == protected_bc:
                # edgeSeg['p1']/['p2'] are plain point ids; *2 converts to coord-array index units
                # that match the values stored in triangulation.triangles (used by splitPass)
                protected_edges.add(frozenset((edgeSeg['p1'] * 2, edgeSeg['p2'] * 2)))
        self.protectedSplitEdges = protected_edges
        return protected_edges

    def count_boundary_nodes(self, bc: int) -> int:
        """Returns the number of mesh edge segments whose geometry bcFlag equals bc."""
        return sum(
            1 for edgeSeg in self.meshObj['edgeSegments']
            if self.in2dObj['segments'][edgeSeg['ednr1']]['bcFlag'] == bc
        )

    def _defaultMesh(in2dObj: Dict) -> Dict:
        """
        Constructs a default mesh from the given In2D object using Delaunay triangulation.

        This function collects mesh points from the In2D segments, builds boundary edge 
        segments, and applies SciPy's Delaunay triangulation to generate triangles. The result 
        is packaged into a dictionary that represents the mesh structure.

        Args:
            in2dObj (Dict): Parsed In2D object containing geometry and boundary segment information.

        Returns:
            meshObj (Dict): A dictionary with keys:
                - 'points': Flattened list of mesh point coordinates [x0, y0, x1, y1, ...].
                - 'triangles': List of triangle vertex indices (flattened, doubled for 2D format).
                - 'edgeSegments': List of dictionaries defining edge segment data for boundary representation.
        """
        # collecting points actually present in the mesh
        meshPoints = []
        points = []
        for i in range(len(in2dObj['segments'])):
            segment = in2dObj['segments'][i]
            meshPoints.append(segment['p1'])
            points.append(in2dObj['points'][segment['p1']*2])
            points.append(in2dObj['points'][segment['p1']*2+1])

        # generating edge segments for mesh 
        boundaryObj = []
        for i in range(len(in2dObj['segments'])):
            segment = in2dObj['segments'][i]

            # finding renumbered indexes according to points in mesh
            if segment['np'] == 2:
                p1 = meshPoints.index(segment['p1'])
                p2 = meshPoints.index(segment['p2'])
            else:
                p1 = meshPoints.index(segment['p1'])
                p2 = meshPoints.index(segment['p3'])

            segObject = {
                'surfid': segment['bcFlag'],
                'p1': p1,
                'p2': p2,
                'sf1': segment['dl'],
                'sf2': segment['dr'],
                'ednr1': i,
                'dist1': 0,
                'ednr2': i,
                'dist2': 1
            }
            boundaryObj.append(segObject)

        # build delaunay triangulation
        xs = points[::2]
        ys = points[1::2]
        coords = []
        for idx in range(len(xs)):
            coords.append([xs[idx], ys[idx]])

        # SciPy Delaunay triangulator
        delTri = Delaunay(coords)
        triangles = [i*2 for row in delTri.simplices for i in row]

        meshObj = {
            'points': points,
            'triangles': triangles,
            'edgeSegments': boundaryObj
        }

        return meshObj
    

