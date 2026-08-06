import numpy as np
import multiprocessing
from typing import TYPE_CHECKING, List, Union, Tuple, Optional
if TYPE_CHECKING:
    from .triangulation import Triangulation
# import numba

# @numba.njit(error_model='numpy')
# def triangleContainingThisPoint(coordinates, triangles, adjacents, initialTriangleIndex, goalPoint, boundary, searchGrid):
#     idx = searchGrid.searchTriangle(goalPoint[0], goalPoint[1])
#     if idx != -1: return idx
#     else: 
#         print(goalPoint, idx)
#         raise ValueError
#     currentTriangleIndex = initialTriangleIndex

#     # to catch cycling due to point on edge
#     prevTriangleIndex = None

#     # while we dont have goal point in current triangle
#     # while not isPointInsideTriangle(triangulation, currentTriangleIndex, goalPoint):
#     # max iterations set to triangle array length as it cannot be greater than this
#     for j in range(len(triangles)):
#         if currentTriangleIndex == None: raise ValueError(goalPoint, nextTriangleIndex, prevTriangleIndex, j)
#         try:
#             isPointInsideTriangle(coordinates, triangles, currentTriangleIndex, goalPoint)
#         except:
#             print(goalPoint)
#         if isPointInsideTriangle(coordinates, triangles, currentTriangleIndex, goalPoint): break
#         # error in finding it
#         if j == len(triangles)-1: 
#             print(coordinates[triangles[currentTriangleIndex]], coordinates[triangles[currentTriangleIndex+1]], coordinates[triangles[currentTriangleIndex+2]])
#             print(coordinates[triangles[currentTriangleIndex]+1], coordinates[triangles[currentTriangleIndex+1]+1], coordinates[triangles[currentTriangleIndex+2]+1])
#             raise ValueError("Search for {} could not be converged with initial triangle {}.".format(goalPoint, initialTriangleIndex))
#         # centroid of current triangle
#         cx = (1/3)*(coordinates[triangles[currentTriangleIndex]]+coordinates[triangles[currentTriangleIndex+1]]+coordinates[triangles[currentTriangleIndex+2]])
#         cy = (1/3)*(coordinates[triangles[currentTriangleIndex]+1]+coordinates[triangles[currentTriangleIndex+1]+1]+coordinates[triangles[currentTriangleIndex+2]+1])

#         # find which edge of current triangle intersects c-goalPoint segment
#         ux = None
#         uy = None
#         vx = None
#         vy = None
#         nextTriangleIndex = None
#         for i in range(3):
#             ux = coordinates[triangles[currentTriangleIndex+i]]
#             uy = coordinates[triangles[currentTriangleIndex+i]+1]
#             vx = coordinates[triangles[currentTriangleIndex+(i+1)%3]]
#             vy = coordinates[triangles[currentTriangleIndex+(i+1)%3]+1]

#             # check if goal point is one of the points of segment
#             EPSILON = 10e-10
#             if (abs(goalPoint[0]-ux) < EPSILON and abs(goalPoint[1]-uy) < EPSILON) or (abs(goalPoint[0]-vx) < EPSILON and abs(goalPoint[1]-vy) < EPSILON):
#                 # if idx != currentTriangleIndex: 
#                 #     print(goalPoint, idx, currentTriangleIndex)
#                 #     raise RuntimeError
#                 return currentTriangleIndex
            
#             if doSegmentsIntersect(cx, cy, goalPoint[0], goalPoint[1], ux, uy, vx, vy):
#                 # choose next triangle adjacent to that intersecting edge
#                 nextTriangleIndex = adjacents[currentTriangleIndex+i]

#                 break
#         if nextTriangleIndex == None: 
#             print(currentTriangleIndex, goalPoint)
#             raise ValueError('here')
#         # # if goal point outside domain, boundary is hit and next adjacent will be -1
#         # if nextTriangleIndex == -1: return currentTriangleIndex

#         # if next triangle is -1, we hit an empty cavity, to get around we set current triangle
#         # index randomly and start searching again
#         # if next triangle is -1
#         if nextTriangleIndex == -1:
#             # if goalPoint is on current triangle
#             if isPointOnTriangle(coordinates, triangles, currentTriangleIndex, goalPoint):
#                 # if idx != currentTriangleIndex: 
#                 #     print(goalPoint, idx, currentTriangleIndex)
#                 #     raise RuntimeError
#                 return currentTriangleIndex
#             else:
#                 # currentTriangleIndex = random.randint(0, len(triangles)//3-1)*3
#                 # 0.42915586335
#                 # find point closest to the goal point on the boundary
#                 # currentTriangleIndex = findClosestTriangle(goalPoint, coordinates, triangles, boundary)
#                 # print(coordinates[triangles[currentTriangleIndex]], goalPoint)
#                 currentTriangleIndex = searchGrid.searchTriangle(goalPoint[0], goalPoint[1])
#                 if currentTriangleIndex == -1: 
#                     print(goalPoint)
#                     raise ValueError
#                 if type(currentTriangleIndex) == list:
#                     currentTriangleIndex = currentTriangleIndex[0]
#                 continue
        

#         # see if cycling of triangle happens because of point on edge
#         if prevTriangleIndex == nextTriangleIndex:
#             # returning both adjacent triangle indexes
#             # if idx != [currentTriangleIndex, prevTriangleIndex] and idx != [prevTriangleIndex, currentTriangleIndex]: 
#             #         print(goalPoint, idx, [currentTriangleIndex, prevTriangleIndex])
#             #         raise RuntimeError
#             return [currentTriangleIndex, prevTriangleIndex]
        
#         prevTriangleIndex = currentTriangleIndex
#         currentTriangleIndex = nextTriangleIndex
    
#     # if idx != currentTriangleIndex: 
#     #     print(goalPoint, idx, currentTriangleIndex)
#     #     raise RuntimeError
#     return currentTriangleIndex

def trianglesTraversal(
        triangulation: 'Triangulation', 
        initialTriIdx: int, 
        goalPoint: List[float]) -> Tuple[bool, Union[int, List[float]]]:
    """
    Traverses through the triangulation to find the triangle containing a given point (goalPoint).

    This function starts from an initial triangle (`initialTriIdx`) and follows adjacent triangles until it finds the 
    triangle that contains the goal point, or determines that the goal point is outside the triangulation. The function 
    handles cases where the goal point lies on the boundary of a triangle or intersects the edges.

    Parameters:
        triangulation (Triangulation): The triangulation object containing the triangles, coordinates, and adjacency information.
        initialTriIdx (int): The index of the starting triangle for the traversal.
        goalPoint (List[float]): The target point [x, y] to search for within the triangles.

    Returns:
        Tuple[bool, Union[int, Tuple[float]]]: A tuple where:
            - The first value is a boolean indicating whether the goal point was found inside a triangle (True) or not (False).
            - The second value is either:
                - The index of the triangle that contains the goal point (if found), or
                - A tuple with the centroid coordinates of the current triangle if the goal point is outside boundary, or
                - A list containing the two triangles' indices if cycling occurs due to the goal point lying exactly on an edge.
    """
    coords = triangulation.coordinates
    triangles = triangulation.triangles
    adjacents = triangulation.adjacents

    currentTriIdx = initialTriIdx

    # to catch cycling due to point on edge
    prevTriIdx = None

    for i in range(len(triangulation.triangles)):
        # print(currentTriIdx)
        if isPointInsideTriangle(coords, triangles, currentTriIdx, goalPoint): 
            return True, currentTriIdx
        # if current triangle on boundary, do additional check
        if adjacents[currentTriIdx] == -1 or adjacents[currentTriIdx+1] == -1 or adjacents[currentTriIdx+2] == -1:
            triIdx = isPointInsideTriangleAvd(triangulation, currentTriIdx, goalPoint[0], goalPoint[1])
            if triIdx != -1:
                return True, triIdx

        # centroid of current triangle
        cx = (1/3)*(coords[triangles[currentTriIdx]]+coords[triangles[currentTriIdx+1]]+coords[triangles[currentTriIdx+2]])
        cy = (1/3)*(coords[triangles[currentTriIdx]+1]+coords[triangles[currentTriIdx+1]+1]+coords[triangles[currentTriIdx+2]+1])

        # find which edge of current triangle intersects c-goalPoint segment
        ux = None
        uy = None
        vx = None
        vy = None
        nextTriangleIndex = None
        for j in range(3):
            ux = coords[triangles[currentTriIdx+j]]
            uy = coords[triangles[currentTriIdx+j]+1]
            vx = coords[triangles[currentTriIdx+(j+1)%3]]
            vy = coords[triangles[currentTriIdx+(j+1)%3]+1]

            # check if goal point is one of the points of segment
            EPSILON = 1e-10 * ((vx-ux)**2 + (vy-uy)**2)    # adaptive epsilon based on edge length scale
            if (abs(goalPoint[0]-ux) < EPSILON and abs(goalPoint[1]-uy) < EPSILON) or (abs(goalPoint[0]-vx) < EPSILON and abs(goalPoint[1]-vy) < EPSILON):
                return True, currentTriIdx
            
            if doSegmentsIntersect(cx, cy, goalPoint[0], goalPoint[1], ux, uy, vx, vy):
                # choose next triangle adjacent to that intersecting edge
                nextTriangleIndex = adjacents[currentTriIdx+j]
                break

        if nextTriangleIndex == None: 
            # print(currentTriIdx, goalPoint)
            # from .triangulationDraw import drawTriangulation
            # import matplotlib.pyplot as plt
            # drawTriangulation(triangulation)
            # plt.show()
            # raise ValueError('here')
            return False, currentTriIdx
        
        if nextTriangleIndex == -1:
            return False, [cx, cy]
        
        # see if cycling of triangle happens because of point on edge
        if prevTriIdx == nextTriangleIndex:
            return True, [currentTriIdx, prevTriIdx]
        
        prevTriIdx = currentTriIdx
        currentTriIdx = nextTriangleIndex
    
    return False, None

def findNearestTriangle(x, y, mesh):
    minDist = 1e20
    bestTri = -1
    
    for i in range(0, len(mesh.triangles), 3):
        n0 = mesh.triangles[i]
        n1 = mesh.triangles[i+1]
        n2 = mesh.triangles[i+2]
        
        cx = (mesh.coordinates[n0] + mesh.coordinates[n1] + mesh.coordinates[n2]) / 3
        cy = (mesh.coordinates[n0+1] + mesh.coordinates[n1+1] + mesh.coordinates[n2+1]) / 3
        
        d = (cx - x)**2 + (cy - y)**2
        
        if d < minDist:
            minDist = d
            bestTri = i
    
    return bestTri

def searchTriangle(x: float, y: float, triangulation: 'Triangulation', startTriIdx: Optional[int]=None) -> int:
    """
    Searches for the triangle that contains the point (x, y) in the given triangulation.

    This function queries the triangulation's element ADT (Alternating Digital
    Tree, `triangulation.elementADT`) for every triangle whose axis aligned
    bounding box contains (x, y), then checks those candidates (smallest
    bounding box first) with an exact point-in-triangle test, falling back to
    a boundary-tolerant test if the point lies exactly on a shared edge or
    vertex.

    Parameters:
        x (float): The x-coordinate of the target point.
        y (float): The y-coordinate of the target point.
        triangulation (Triangulation): The triangulation object containing the points and triangles.
        startTriIdx (Optional[int]): Unused. Present for backwards
            compatibility with callers that pass a starting triangle index.

    Returns:
        int: The index of the triangle containing the point (x, y), or -1 if the point is outside the triangulation.
    """
    return triangulation.elementADT.searchElement(x, y, triangulation)

# def findClosestTriangle(goalPoint, coords, triangles, boundary):

#     dMin = np.inf
#     closestPoint = None

#     # for all boundary points
#     for i in range(len(boundary)):
#         xp = coords[boundary[i]*2]
#         yp = coords[boundary[i]*2+1]
#         xgp = goalPoint[0]
#         ygp = goalPoint[1]

#         # find dist
#         d = (xp-xgp)*(xp-xgp) + (yp-ygp)*(yp-ygp)

#         if d < dMin:
#             # set current point as closest point
#             closestPoint = boundary[i]*2
#             dMin = d
    
#     # find circling triangles at this point
#     circTris = trianglesHavingThisNode(triangles, closestPoint)

#     return circTris[0]

def isPointInsideTriangle(coordinates: List[float], triangles: List[int], triangleIndex: int, point: List[float]) -> bool:
    """
    Checks if a given point is inside the specified triangle in the triangulation.

    This function works by checking the orientation of the point with respect to each edge of the triangle. 
    If the point lies on the same side for all three edges, it is inside the triangle.

    Parameters:
        coordinates (List[float]): The list of coordinates of the points in the triangulation.
        triangles (List[int]): The list of triangles, each defined by three indices into the coordinates list.
        triangleIndex (int): The index of the triangle to check.
        point (List[float]): A list containing the x and y coordinates of the point to check.

    Returns:
        bool: `True` if the point is inside the triangle, `False` otherwise.
    """

    count = 0

    for i in range(3):
        ux = coordinates[triangles[triangleIndex+i]]
        uy = coordinates[triangles[triangleIndex+i]+1]
        vx = coordinates[triangles[triangleIndex+(i+1)%3]]
        vy = coordinates[triangles[triangleIndex+(i+1)%3]+1]

        if isCcwOrientation(ux, uy, vx, vy, point[0], point[1]) == 1:
            count+=1

    return count == 3

def isPointOnTriangle(coordinates: List[float], triangles: List[int], triangleIndex: int, point: List[float]) -> bool:
    """
    Checks if a given point lies on the edge or any of the corners of the specified triangle.

    This function works by calculating the cross-product for each edge of the triangle. 
    If the cross-product is close to zero (within a small `EPSILON` threshold), the point is considered to lie on the edge.

    Parameters:
        coordinates (List[float]): The list of coordinates of the points in the triangulation.
        triangles (List[int]): The list of triangles, each defined by three indices into the coordinates list.
        triangleIndex (int): The index of the triangle to check.
        point (List[float]): A list containing the x and y coordinates of the point to check.

    Returns:
        bool: `True` if the point is on the edge or any of the corners of the triangle, `False` otherwise.
    """
    count = 0

    for i in range(3):
        ux = coordinates[triangles[triangleIndex+i]]
        uy = coordinates[triangles[triangleIndex+i]+1]
        vx = coordinates[triangles[triangleIndex+(i+1)%3]]
        vy = coordinates[triangles[triangleIndex+(i+1)%3]+1]

        val = (ux-point[0])*(vy-point[1]) - (uy-point[1])*(vx-point[0])

        # adaptive tolerance matching isCcwOrientation's collinearity check
        # for this edge, so any edge it treats as collinear with `point` is
        # also treated as "on" here
        axN = ux-point[0]
        ayN = uy-point[1]
        bxN = vx-point[0]
        byN = vy-point[1]
        EPSILON = 1e-10 * max(axN*axN + ayN*ayN, bxN*bxN + byN*byN)

        if abs(val) < EPSILON:
            count+=1

    return count != 0

def isPointInsideTriangleAvd(triangulation: 'Triangulation', triIdx: int, x: float, y: float) -> Union[int, List[int]]:
    """
    Sophisticated check to determine if a point is inside a triangle or not.

    This function uses several geometric checks to determine if the given point (x, y) lies 
    inside the triangle indexed by `triIdx` in the triangulation. The check includes orientation 
    tests, segment intersection checks, and proximity checks for vertices and edges.

    Parameters:
        triangulation: The triangulation object that contains information about the coordinates, 
                        triangles, and adjacent triangles.
        triIdx (int): The index of the triangle to check.
        x (float): The x-coordinate of the point to check.
        y (float): The y-coordinate of the point to check.

    Returns:
        Union[int, List[int]]: 
            - Returns the index of the triangle if the point is inside or on the triangle.
            - If the point is on the boundary, returns a list of the current triangle index and 
              the adjacent triangle index.
            - Returns -1 if the point is outside the triangle.
    
    """
    coords = triangulation.coordinates
    triangles = triangulation.triangles
    adjacents = triangulation.adjacents
    
    p1x = coords[triangles[triIdx]]
    p1y = coords[triangles[triIdx]+1]
    p2x = coords[triangles[triIdx+1]]
    p2y = coords[triangles[triIdx+1]+1]
    p3x = coords[triangles[triIdx+2]]
    p3y = coords[triangles[triIdx+2]+1]

    # orientations
    o1 = isCcwOrientation(p1x, p1y, p2x, p2y, x, y)
    o2 = isCcwOrientation(p2x, p2y, p3x, p3y, x, y)
    o3 = isCcwOrientation(p3x, p3y, p1x, p1y, x, y)

    # if anyone orientation is zero and point-opposite vertex segment intersects, point is on triangle
    if o1 == 0 and doSegmentsIntersect(p1x, p1y, p2x, p2y, x, y, p3x, p3y):
        adj = adjacents[triIdx]
        if adj == -1: return triIdx # if on boundary
        return [triIdx, adj]
    if o2 == 0 and doSegmentsIntersect(p2x, p2y, p3x, p3y, x, y, p1x, p1y):
        adj = adjacents[triIdx+1]
        if adj == -1: return triIdx # if on boundary
        return [triIdx, adj]
    if o3 == 0 and doSegmentsIntersect(p3x, p3y, p1x, p1y, x, y, p2x, p2y):
        adj = adjacents[triIdx+2]
        if adj == -1: return triIdx # if on boundary
        return [triIdx, adj]
    
    # if any of the triangle point is in close proximity to the given point
    EPSILON = 10e-10
    if abs(p1x-x) < EPSILON and abs(p1y-y) < EPSILON \
        or abs(p2x-x) < EPSILON and abs(p2y-y) < EPSILON \
        or abs(p3x-x) < EPSILON and abs(p3y-y) < EPSILON:
        return triIdx
    
    # if point is in close proximity of one of the edges
    EPSILON = triangulation.epsilon

    # check whether point projetion lies on the segment
    if triangulation.adjacents[triIdx] == -1:
        denom1 = (p2x-p1x)**2 + (p2y-p1y)**2
        t1 =  ((x-p1x)*(p2x-p1x) + (y-p1y)*(p2y-p1y))/denom1
        if 0 <= t1 <= 1:
            d1 = abs((p2y-p1y)*x - (p2x-p1x)*y + p2x*p1y - p2y*p1x)/(denom1)**0.5
            if d1 < EPSILON: return triIdx

    if triangulation.adjacents[triIdx+1] == -1:
        denom2 = (p3x-p2x)**2 + (p3y-p2y)**2
        t2 =  ((x-p2x)*(p3x-p2x) + (y-p2y)*(p3y-p2y))/denom2
        if 0 <= t2 <= 1:
            d2 = abs((p3y-p2y)*x - (p3x-p2x)*y + p3x*p2y - p3y*p2x)/(denom2)**0.5
            if d2 < EPSILON: return triIdx
    
    if triangulation.adjacents[triIdx+2] == -1:
        denom3 = (p1x-p3x)**2 + (p1y-p3y)**2
        t3 =  ((x-p3x)*(p1x-p3x) + (y-p3y)*(p1y-p3y))/denom3
        if 0 <= t3 <= 1:
            d3 = abs((p1y-p3y)*x - (p1x-p3x)*y + p1x*p3y - p1y*p3x)/(denom3)**0.5
            if d3 < EPSILON: return triIdx
    
    return -1

def adjacencyList(triangles: List[int], nPoints: int) -> List[int]:
    """
    Establish adjacency relationships between triangles in a triangulation.

    This function takes an array of triangle indices and the total number of points in the triangulation,
    and returns a list representing adjacency relationships between the triangles. Each entry in the returned list
    represents the adjacent triangle (or -1 if there is no adjacency).

    The adjacency relationships are determined based on shared edges between triangles. 

    Parameters:
        triangles (List[int]): A list of triangle indices. Each triangle is represented by a list of three point indices.
        nPoints (int): The total number of points in the triangulation.

    Returns:
        List[int]: A list where each index corresponds to a triangle edge, and the value at that index is the adjacent triangle's index
                   to that edge. If no adjacent triangle exists, the value will be -1.

    Reference:
        Book: "Meshing, Geometric Modeling and Numerical Simulation 1", sections 4.4 and 9.4.1
    """
    # hash table to store edges with same keys
    hashTable = [0]*nPoints

    # initialize adjacents
    adjacents = [-1]*len(triangles)

    # for all triangles
    for triIdx in range(0, len(triangles), 3):
        # for all three edges in this triangle
        for edge in range(3):
            u = triangles[triIdx+edge]//2
            v = triangles[triIdx+(edge+1)%3]//2

            # key to index in hash table
            key = (u+v)%nPoints

            # if new edge at this key, store in hash table
            if hashTable[key] == 0:
                hashTable[key] = [(triIdx, edge, min(u, v))]
            # already encountered edge in this key
            else:
                # loop through previous edges at this key
                adjEdge = None
                for hashEdge in hashTable[key]:
                    # if failsafe value matches
                    if hashEdge[2] == min(u, v):
                        # found adjacent edge  
                        adjEdge = hashEdge
                        break
                
                # if found adjacent edge part, establish adjacency relationships
                if adjEdge:
                    adjTriIdx = adjEdge[0]
                    adjEdgeIdx = adjEdge[1]
                    adjacents[triIdx+edge] = adjTriIdx
                    adjacents[adjTriIdx+adjEdgeIdx] = triIdx
                else:
                    # add new edge at same key with different failsafe value
                    hashTable[key].append((triIdx, edge, min(u, v)))

    return adjacents

def ballOfNode(triangulation: 'Triangulation', node: int) -> List[int]:
    """
    Constructs a ball of triangles around a given node in the triangulation.

    This function computes the set of triangles that form a "ball" around the given node, 
    meaning all triangles that share at least one vertex with the given node. It traverses 
    the triangulation's adjacency relationships to find all the neighboring triangles connected 
    to the node, moving forward (CCW) and backward (CW) through the triangulation.

    Parameters:
        triangulation (Triangulation): The triangulation object containing triangles and adjacency relationships.
        node (int): The index of the node for which the ball of triangles is to be constructed. 

    Returns:
        List[int]: A list of triangle indices that form the ball around the given node.
    
    Raises:
        ValueError: If the traversal exceeds the maximum allowed iterations (500).
    
    Reference:
        Book: "Meshing, Geometric Modeling and Numerical Simulation 1", section 9.4.2 and algo 9.19
    """
    triangles = triangulation.triangles
    adjacents = triangulation.adjacents

    # seed triangle to start with
    seedTriIdx = triangulation.adjPointTris[node//2]
    currentTriIdx = seedTriIdx

    ballTriangles = [seedTriIdx]

    forwardPass = True
    
    for i in range(50000):
        if i == 49999: raise ValueError

        # find local index of node in triangle
        nodeIdx = [triangles[currentTriIdx], triangles[currentTriIdx+1], triangles[currentTriIdx+2]].index(node)
        if forwardPass:
            # finding forward adjacent
            adjTriIdx = adjacents[currentTriIdx+(nodeIdx-1)%3]
        else:
            # finding backward adjacent
            adjTriIdx = adjacents[currentTriIdx+(nodeIdx)%3]

        # if hit boundary, start backward pass from seed triangle
        if adjTriIdx == -1 and forwardPass:
            forwardPass = False
            currentTriIdx = seedTriIdx
            continue

        # if hit boundary in backward pass
        if adjTriIdx == -1 and not forwardPass:
            break

        # if completed ball around node
        if adjTriIdx == seedTriIdx: break

        # add this new triangle
        ballTriangles.append(adjTriIdx)
        # move to adjacent triangle
        currentTriIdx = adjTriIdx
    
    return ballTriangles

def collectEdges(triangulation: 'Triangulation') -> List[int]:
    """
    Constructs an edge list from the triangulation data.

    This function iterates over the triangles in the triangulation and constructs a list of edges 
    by collecting the indices of the endpoints in the coordinates array. It ensures that each edge 
    is stored only once, even if it appears multiple times across different triangles, by using a 
    hash table to detect duplicate edges.

    Parameters:
        triangulation (Triangulation): The triangulation object containing triangles and the coordinates.

    Returns:
        List[int]: A list of global indices that represent edges. Each element in the list corresponds 
                   to an edge in triangles list and is defined by two points in the coordinates array.

    Reference:
        Book: "Meshing, Geometric Modeling and Numerical Simulation 1", section 4.4
    """
    # stores idxs of endpoints in coords array defining an edge
    edges = []
    
    # hash table to store edges with same keys
    nPoints = len(triangulation.coordinates)//2
    hashTable = [0]*nPoints
    lenTriangles = len(triangulation.triangles)
    for i in range(0, lenTriangles, 3):
        # for all three edges
        for j in range(3):
            # # skip edges on boundary hull
            # if triangulation.adjacents[i+j] == -1: continue

            u = triangulation.triangles[i+j]
            v = triangulation.triangles[i+(j+1)%3]
            uIdx = i+j
            # key to index in hash table
            key = (u+v)%nPoints

            # if new edge at this key, store in hash table
            if hashTable[key] == 0:
                hashTable[key] = [(i, j, min(u, v))]
                # also add global index to edge list
                edges.append(uIdx)
            else: # already encountered edge in this key
                # loop through previous edges at this key
                sameEdge = False
                for hashEdge in hashTable[key]:
                    # if failsafe value matches
                    if hashEdge[2] == min(u, v):
                        # found same edge  
                        sameEdge = True
                        break
                
                if not sameEdge:
                    # add new edge at same key with different failsafe value
                    hashTable[key].append((i, j, min(u, v)))
                    # also add global index to edge list
                    edges.append(uIdx)
    
    return edges

# @numba.njit(error_model='numpy')
# def isCcwOrientation(ax, ay, bx, by, cx, cy):
#     EPSILON = 1e-14
#     val = (ax-cx)*(by-cy) - (ay-cy)*(bx-cx)

#     if abs(val) < EPSILON:
#         # collinear
#         return 0
#     elif val > EPSILON:
#         # CCW
#         return 1
#     elif val < -EPSILON:
#         # CW
#         return -1
    
def isCcwOrientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> int:
    """
    Determines the orientation of the triplet of points (a, b, c) in 2D space.

    This function computes the cross product of vectors AB and AC to determine if the 
    triplet of points (a, b, c) forms a counter-clockwise (CCW), clockwise (CW), or 
    collinear arrangement. It returns:
    - 1 for counter-clockwise,
    - -1 for clockwise,
    - 0 for collinear.

    The function adapts the epsilon tolerance based on the magnitude of the vectors to 
    handle small geometric precision errors.

    Parameters:
        ax (float): x-coordinate of point A.
        ay (float): y-coordinate of point A.
        bx (float): x-coordinate of point B.
        by (float): y-coordinate of point B.
        cx (float): x-coordinate of point C.
        cy (float): y-coordinate of point C.

    Returns:
        int: 1 if counter-clockwise, -1 if clockwise, 0 if collinear.
    """
    EPSILON = 1e-10
    axN = ax-cx
    ayN = ay-cy
    bxN = bx-cx
    byN = by-cy

    # val = (ax-cx)*(by-cy) - (ay-cy)*(bx-cx)
    val = axN*byN - ayN*bxN

    magSqr = max(axN**2 + ayN**2, bxN**2 + byN**2)

    EPSILON *= magSqr   # adaptive epsilon based on edge length scale

    if abs(val) < EPSILON:
        # collinear
        return 0
    elif val > EPSILON:
        # CCW
        return 1
    elif val < -EPSILON:
        # CW
        return -1

# import ctypes
# # Load the shared library (since it's in the same folder as the Python script)
# predicate = ctypes.CDLL('/home/priyansu/project/meshAdapt/libpredicates.so') 

# # Define the argument types and return type for the orient2d function
# predicate.orient2d.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# predicate.orient2d.restype = ctypes.c_double

# # Prepare input data (REAL* pa, pb, pc)
# pa = (ctypes.c_double * 2)()
# pb = (ctypes.c_double * 2)()
# pc = (ctypes.c_double * 2)()

# def isCcwOrientation(ax, ay, bx, by, cx, cy):

#     # Update the values of the reusable arrays
#     pa[0], pa[1] = ax, ay
#     pb[0], pb[1] = bx, by
#     pc[0], pc[1] = cx, cy

#     val = predicate.orient2d(pa, pb, pc)

#     if val == 0:
#         # collinear
#         return 0
#     elif val > 0:
#         # CCW
#         return 1
#     elif val < 0:
#         # CW
#         return -1

def isOnSegment(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
    """
    Checks if point B lies on the line segment defined by points A and C, assuming 
    that A, B, and C are collinear.

    This function verifies whether point B is within the bounding box of the segment AC,
    which ensures that B lies on the line segment between A and C, but not beyond it.

    Parameters:
        ax (float): x-coordinate of point A.
        ay (float): y-coordinate of point A.
        bx (float): x-coordinate of point B.
        by (float): y-coordinate of point B.
        cx (float): x-coordinate of point C.
        cy (float): y-coordinate of point C.

    Returns:
        bool: True if point B lies on the segment AC, False otherwise.
    """
    if (bx <= max(ax, cx)) and (bx >= min(ax, cx)) and (by <= max(ay, cy)) and (by >= min(ay, cy)):
        return True
    return False

def doSegmentsIntersect(ax0: float, ay0: float, ax1: float, ay1: float, bx0: float, by0: float, bx1: float, by1: float) -> bool:
    """
    Determines whether two line segments (A-B and C-D) intersect.

    This function uses orientation tests and checks whether the segments are collinear 
    and overlap. It returns True if the segments intersect, otherwise False.

    Parameters:
        ax0 (float): x-coordinate of the first point of segment A.
        ay0 (float): y-coordinate of the first point of segment A.
        ax1 (float): x-coordinate of the second point of segment A.
        ay1 (float): y-coordinate of the second point of segment A.
        bx0 (float): x-coordinate of the first point of segment B.
        by0 (float): y-coordinate of the first point of segment B.
        bx1 (float): x-coordinate of the second point of segment B.
        by1 (float): y-coordinate of the second point of segment B.

    Returns:
        bool: True if the segments intersect, False otherwise.
    """
    
    o1 = isCcwOrientation(ax0, ay0, ax1, ay1, bx0, by0)
    o2 = isCcwOrientation(ax0, ay0, ax1, ay1, bx1, by1)
    o3 = isCcwOrientation(bx0, by0, bx1, by1, ax0, ay0)
    o4 = isCcwOrientation(bx0, by0, bx1, by1, ax1, ay1)
    
    if ((o1 != o2) and (o3 != o4)): return True

    if (o1 == 0 and isOnSegment(ax0, ay0, bx0, by0, ax1, ay1)): return True
    if (o2 == 0 and isOnSegment(ax0, ay0, bx1, by1, ax1, ay1)): return True
    if (o3 == 0 and isOnSegment(bx0, by0, ax0, ay0, bx1, by1)): return True
    if (o4 == 0 and isOnSegment(bx0, by0, ax1, ay1, bx1, by1)): return True

    return False

# find intersection of given segments a dna b end point coordinates
def segmentIntersection(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
    # using Cramer's rule
    # denoinator determenant
    denom = (ax0 - ax1) * (by0 - by1) - (ay0 - ay1) * (bx0 - bx1)
    
    # check if parallel lines
    if denom == 0:
        # raise RuntimeError("Parallel lines!")
        return None

    # x and y coordinates of the intersection point
    x = ((ax0 * ay1 - ay0 * ax1) * (bx0 - bx1) - (ax0 - ax1) * (bx0 * by1 - by0 * bx1)) / denom
    y = ((ax0 * ay1 - ay0 * ax1) * (by0 - by1) - (ay0 - ay1) * (bx0 * by1 - by0 * bx1)) / denom

    return [x, y]

def isOnSegment(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
    """
    Checks if the point (bx, by) lies on the line segment defined by points (ax, ay) and (cx, cy).

    The function first checks if the point (bx, by) lies within the bounding box of the segment 
    defined by (ax, ay) and (cx, cy). If this is true, it then checks for collinearity using 
    the cross product (which corresponds to the area of the triangle formed by the points).

    Parameters:
        ax (float): x-coordinate of point A (first endpoint of the segment).
        ay (float): y-coordinate of point A.
        bx (float): x-coordinate of point B (the point to check).
        by (float): y-coordinate of point B.
        cx (float): x-coordinate of point C (second endpoint of the segment).
        cy (float): y-coordinate of point C.

    Returns:
        bool: True if point (bx, by) lies on the segment from (ax, ay) to (cx, cy), otherwise False.
    """
    # check if (bx, by) is within the bounding box of the segment (ax, ay) -> (cx, cy)
    if (bx <= max(ax, cx)) and (bx >= min(ax, cx)) and (by <= max(ay, cy)) and (by >= min(ay, cy)):
        # Check for collinearity by using the cross product (Area of triangle should be 0)
        if (cy - ay) * (bx - cx) == (by - cy) * (cx - ax):
            return True
    return False

def isPointInsidePolygon(point: List[float], coords: List[float], boundary: List[int]) -> bool:
    """
    Determines if a given point is inside a polygon using the ray-casting algorithm.

    The function casts a ray from the given point in the positive x-direction and counts how many 
    times it intersects the edges of the polygon. If the count is odd, the point is inside the polygon; 
    otherwise, it is outside.

    Parameters:
        point (List[float]): A list representing the coordinates of the point to check, [x, y].
        coords (List[float]): A list of coordinates for all the points in the polygon, where each pair of 
                              consecutive values represents a point (x, y).
        boundary (List[int]): A list of indices that represent the vertices of the polygon. Each value in the list 
                               corresponds to the index of a point in the `coords` list.

    Returns:
        bool: True if the point is inside the polygon, False if it is outside.
    
    Reference:
        Webpage: https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.htmlclear
    """
    counter = 0
    N = len(boundary)
    # print(type(boundary[0]), boundary[0])
    for i in range(0, N, 2):
        p1 = [coords[boundary[i]], coords[boundary[i]+1]]
        p2 = [coords[boundary[i+1]], coords[boundary[i+1]+1]]
        
        if isOnSegment(p1[0], p1[1], point[0], point[1], p2[0], p2[1]):
            return True
        
        if point[1] > min(p1[1], p2[1]):
            if point[1] <= max(p1[1], p2[1]):
                if point[0] <= max(p1[0], p2[0]):
                    if p1[1] != p2[1]:
                        xInters = (point[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                        if p1[0] == p2[0] or point[0] <= xInters:
                            counter += 1

        p1 = p2
    return counter % 2 == 1

def closestBoundaryPoint(boundary, coords, x, y):
    minDist = 1e20
    best = (x, y)

    for edge in boundary:
        # correct keys for YOUR data
        n1 = edge['p1']
        n2 = edge['p2']

        x1, y1 = coords[n1], coords[n1+1]
        x2, y2 = coords[n2], coords[n2+1]

        dx, dy = x2 - x1, y2 - y1
        if dx*dx + dy*dy == 0:
            continue

        # projection
        t = ((x-x1)*dx + (y-y1)*dy)/(dx*dx + dy*dy)
        t = max(0, min(1, t))

        px = x1 + t*dx
        py = y1 + t*dy

        d = (px-x)**2 + (py-y)**2
        if d < minDist:
            minDist = d
            best = (px, py)

    return best

def boundaryProject(boundary: List[int], coords: List[float], segStart: List[float], segEnd: List[float], cast: Optional[bool]=False) -> List[float]:
    """
    Projects a segment onto a polygon boundary and returns the closest point on the boundary segment 
    where the given segment intersects or is clipped by the boundary.

    Parameters:
        boundary (List[int]): A list of indices defining the boundary of the polygon.
        coords (List[float]): A list of coordinates representing the vertices of the polygon.
        segStart (List[float]): A list representing the starting point [x, y] of the segment.
        segEnd (List[float]): A list representing the ending point [x, y] of the segment.
        cast (bool): If True, the function will cast the segment towards the boundary (instead of finding the intersection).

    Returns:
        List[float]: The [x, y] coordinates of the intersection or projected point on the boundary.
    """
    
    # check where the vector is located
    isEndInside = isPointInsidePolygon(segEnd, coords, boundary)

    # find closest clipped point on intersection of boundary and vector
    xMin = None
    yMin = None
    rMin = np.inf
    # for all boundary segments
    for i in range(0, len(boundary), 2):
        bdSegStart = [coords[boundary[i]], coords[boundary[i]+1]]
        bdSegEnd = [coords[boundary[i+1]], coords[boundary[i+1]+1]]

        # if segStart is on bdSegment, continue
        if isOnSegment(bdSegStart[0], bdSegStart[1], segStart[0], segStart[1], bdSegEnd[0], bdSegEnd[1]):
            continue
        
        if doSegmentsIntersect(segStart[0], segStart[1], segEnd[0], segEnd[1], bdSegStart[0], bdSegStart[1], bdSegEnd[0], bdSegEnd[1]) or cast:
            # try finding intersection
            ax0 = segStart[0]
            ay0 = segStart[1]
            ax1 = segEnd[0]
            ay1 = segEnd[1]
            bx0 = bdSegStart[0]
            by0 = bdSegStart[1]
            bx1 = bdSegEnd[0]
            by1 = bdSegEnd[1]

            # denoinator determenant
            # denom = (segStart[0] - segEnd[0]) * (bdSegStart[1] - bdSegEnd[1]) - (segStart[1] - segEnd[1]) * (bdSegStart[0] - bdSegEnd[0])
            denom = (ax0 - ax1) * (by0 - by1) - (ay0 - ay1) * (bx0 - bx1)
            # if denom zero, lines are parallel with nonzero offset, no intersection
            EPS = 1e-14
            if abs(denom) < EPS:
                continue

            # x and y coordinates of the intersection point
            x = ((ax0 * ay1 - ay0 * ax1) * (bx0 - bx1) - (ax0 - ax1) * (bx0 * by1 - by0 * bx1)) / denom
            y = ((ax0 * ay1 - ay0 * ax1) * (by0 - by1) - (ay0 - ay1) * (bx0 * by1 - by0 * bx1)) / denom
            # return [round(x, 15), round(y, 15)]
            r = (x - segStart[0])**2 + (y-segStart[1])**2

            # check intersection is valid point in the forward direction
            if cast:
                dotProd = (segEnd[0]-segStart[0]) * (x-segStart[0]) + (segEnd[1]-segStart[1]) * (y-segStart[1])
                if dotProd < 0: continue

                if abs(bdSegStart[0] - bdSegEnd[0]) > 1e-10:
                    if not (min(bdSegStart[0], bdSegEnd[0]) <= x <= max(bdSegStart[0], bdSegEnd[0])): continue

                if abs(bdSegStart[1] - bdSegEnd[1]) > 1e-10:
                    if not (min(bdSegStart[1], bdSegEnd[1]) <= y <= max(bdSegStart[1], bdSegEnd[1])): continue
                      
            if (r < rMin):
                xMin = x
                yMin = y
                rMin = r
    
    # if segEnd is outside
    if not isEndInside:
        # if no intersection found vector must be starting on boundary and pointing outside
        if xMin == None: return segStart
    # if end is inside
    else:
        # if no intersection found
        if xMin == None: return segEnd

    return [round(xMin, 15), round(yMin, 15)]


