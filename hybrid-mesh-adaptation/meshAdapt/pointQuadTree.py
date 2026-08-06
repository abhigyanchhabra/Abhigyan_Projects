import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import TYPE_CHECKING, Tuple, List, Optional
if TYPE_CHECKING:
    from .triangulation import Triangulation

from .metric import unitVec

# fig, ax = plt.subplots()

class PointQuadTree:
    """
    A class representing a Quadtree data structure for efficiently storing and querying points in 2D space.

    Attributes:
        bounds (tuple): A tuple of the form (minX, minY, maxX, maxY) representing the bounds of the current node.
        capacity (int): The maximum number of points a node can hold before subdividing into child nodes.
        globalCoordinates (list): A list containing the global coordinates of points.
        points (list): A list of point indexes in globalCoordinates contained within the current node.
        divided (bool): A flag indicating whether the node has been subdivided into child nodes.
        northEast (PointQuadTree): The northeastern child node.
        northWest (PointQuadTree): The northwestern child node.
        southEast (PointQuadTree): The southeastern child node.
        southWest (PointQuadTree): The southwestern child node.
        father (PointQuadTree or None): The parent node of the current node.
    """
    def __init__(
            self, 
            bounds: Tuple[float, float, float, float], 
            globalCoordinates: List[float], 
            capacity: Optional[int]=4) -> None:
        self.bounds = bounds
        self.capacity = capacity
        self.globalCoordinates = globalCoordinates
        self.points = []
        self.divided = False
        self.northEast = None
        self.northWest = None
        self.southEast = None
        self.southWest = None
        self.father = None

        # rect = Rectangle((self.bounds[0], self.bounds[1]), abs(self.bounds[0]-self.bounds[2]), abs(self.bounds[1]-self.bounds[3]), linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
    
    def _contains(self, x: float, y: float) -> bool:
        """
        Checks if given point is contained in node bounds
        """
        # node bounds
        minX, minY, maxX, maxY = self.bounds
        
        if (minX <= x < maxX and minY <= y < maxY): return True

        return False

    # inserts point given by idx in global coords array
    def insert(self, point: List[float]) -> bool:
        """
        Inserts a point (given by its index in the globalCoordinates list) into the Quadtree.
        If the node is a leaf and has enough capacity, the point is added directly to the node.
        If the node exceeds its capacity, it subdivides and the point is inserted into one of the child nodes.

        Parameters:
        point (List[float]): A list representing the point to insert. The list contains the indices
                             of the point to insert in the globalCoordinates array.

        Returns:
            bool: True if the point was successfully inserted, False if the point is outside the bounds of the node.
        """
        x = self.globalCoordinates[point]
        y = self.globalCoordinates[point+1]

        if not self._contains(x, y):
            return False
        
        # if leaf node can take this node
        if len(self.points) < self.capacity and not self.divided:
            self.points.append(point)
            return True

        if not self.divided:
            self._subdivide()

        return (self.northEast.insert(point) 
                or self.northWest.insert(point)
                or self.southEast.insert(point)
                or self.southWest.insert(point))

    def _subdivide(self) -> None:
        """
        Subdivides the current Quadtree node into four child nodes (quadrants).
        Each child node represents a quadrant of the current node's bounds. 
        The method also moves all points from the current node into the appropriate child node.

        Parameters:
            None

        Returns:
            None
        """
        # node bounds
        minX, minY, maxX, maxY = self.bounds

        midX = 0.5*(minX+maxX)
        midY = 0.5*(minY+maxY)

        # define quadrants
        quadrants = [
            (minX, minY, midX, midY),   # SW
            (midX, minY, maxX, midY),   # SE
            (minX, midY, midX, maxY),   # NW
            (midX, midY, maxX, maxY)    # NE
        ]

        self.southWest = PointQuadTree(quadrants[0], self.globalCoordinates, capacity=self.capacity)
        self.southEast = PointQuadTree(quadrants[1], self.globalCoordinates, capacity=self.capacity)
        self.northWest = PointQuadTree(quadrants[2], self.globalCoordinates, capacity=self.capacity)
        self.northEast = PointQuadTree(quadrants[3], self.globalCoordinates, capacity=self.capacity)

        # set this node as father of all children
        self.southWest.father = self
        self.southEast.father = self
        self.northWest.father = self
        self.northEast.father = self

        self.divided = True

        # move existing points to children
        for p in self.points:
            self.southWest.insert(p)
            self.southEast.insert(p)
            self.northWest.insert(p)
            self.northEast.insert(p)

        # clear current node points
        self.points = []

    def _findLeafNode(self, x: float, y: float) -> 'Optional[PointQuadTree]':
        """
        Recursively searches for the leaf node that contains the point (x, y).

        This method checks whether the current node has been subdivided. If not, it returns the current node.
        If the node is subdivided, it recursively searches the appropriate child node that contains the point.

        Args:
            x (float): The x-coordinate of the point to search for.
            y (float): The y-coordinate of the point to search for.

        Returns:
            Optional[PointQuadTree]: The leaf node containing the point, or None if no such leaf node is found.
        """
        # found leaf node
        if not self.divided:
            return self

        for child in (self.southWest, self.southEast, self.northWest, self.northEast):
            if child._contains(x, y):
                leaf = child._findLeafNode(x, y)
                if leaf:
                    return leaf

    def _distToBounds(self, x: float, y: float) -> float:
        """
        Returns the squared distance from a point (x, y) to the bounds of the current node.

        Parameters:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.

        Returns:
            float: The squared distance from the point to the bounds of the node.
        """
        # node bounds
        minX, minY, maxX, maxY = self.bounds

        dx = max(minX-x, 0, x-maxX)
        dy = max(minY-y, 0, y-maxY)

        # Warning: returns squared disance for performance
        return dx*dx + dy*dy
    
    # checks if the point is in the given quadrant or not
    def _isPointInQuadrant(self, x, y, px, py, quadrant):
        """
                ^     p.
                |
           1   x|    0
         -------+------->
                |
           2    |    3
                |
        Quadrant mapping:
        dx  dy  q
        +   +   0
        -   +   1
        -   -   2
        +   -   3
        """
        dx = px-x
        dy = py-y
        # extremely fast bit operations to evaluate quadrant based on dx dy signs 
        q = ((dy<0)<<1)|(dx<0)  # results in 0-3

        return q == quadrant

    def _nearestPoint(self, x: float, y: float, node: 'PointQuadTree', minDist: float, nearestPoint: int, searchQuadrant=None) -> Tuple[float, int]:
        """
        Recursively searches for the nearest point to the given point (x, y) within the specified node and its children.

        Parameters:
            x (float): The x-coordinate of the point to search for.
            y (float): The y-coordinate of the point to search for.
            node (PointQuadTree): The current node to search for the nearest point.
            minDist (float): The current minimum distance found in the search.
            nearestPoint (int): The index of the currently nearest point.

        Returns:
            Tuple[float, int]: A tuple containing the minimum distance and the index of the nearest point.
                                        If no point is found, the index will be None.
    
        """
        if not node:
            return minDist, nearestPoint
        
        # if this node is farther than best distance, skip it
        distToBounds = node._distToBounds(x, y)
        if distToBounds > minDist:
            return minDist, nearestPoint
        
        for point in node.points:
            px = node.globalCoordinates[point]
            py = node.globalCoordinates[point+1]

            # if quadrant given and point not in given quadrant, skip
            if searchQuadrant and not node._isPointInQuadrant(x, y, px, py, searchQuadrant):
                continue

            dist = (x-px)**2 + (y-py)**2
            if dist < minDist:
                minDist = dist
                nearestPoint = point  
        
        # if node is subdivided, recursively search children
        if node.divided:
            # calculate distances to each quadrant
            quadrants = [
                (node.southWest, node.southWest._distToBounds(x, y)),
                (node.southEast, node.southEast._distToBounds(x, y)),
                (node.northWest, node.northWest._distToBounds(x, y)),
                (node.northEast, node.northEast._distToBounds(x, y))
            ]

            # sorts quadrants by distance to query point
            quadrants.sort(key=lambda q: q[1])

            # search quadrants in order of proximity
            for quadrant, _ in quadrants:
                minDist, nearestPoint = self._nearestPoint(x, y, quadrant, minDist, nearestPoint, searchQuadrant)

        return minDist, nearestPoint
    
    def searchNearest(self, x: float, y: float, quadrant=None) -> Optional[int]:
        """
        Searches for the nearest point to the given (x, y) coordinates in the Quadtree.

        This method initializes the search for the nearest point starting from the root node.
        It returns the index of the nearest point in the global coordinates array.

        Parameters:
            x (float): The x-coordinate of the point to search for.
            y (float): The y-coordinate of the point to search for.

        Returns:
            Optional[int]: The index of the nearest point in the global coordinates array, or None if no point is found.
        """
        # initialize min distance and nearest point
        minDist = float('inf')
        nearestPoint = None

        # start search from root
        minDist, nearestPoint = self._nearestPoint(x, y, self, minDist, nearestPoint, quadrant)

        return nearestPoint
    
    def closestNodeInRay(self, point: List[float], direction: List[float], maxLength: float=float('inf')) -> 'Optional[PointQuadTree]':
        """
        Finds the closest node intersected by a ray starting from a point and traveling in a given direction.

        The ray is defined by a starting point and a direction vector. The method checks for intersections
        between the ray and the node's bounds and returns the closest node it intersects within a maximum length.

        Parameters:
            point (List[float]): The starting point of the ray, represented as a list of two floats [x, y].
            direction (List[float]): The direction vector of the ray, represented as a list of two floats [dx, dy].
            maxLength (float): The maximum length of the ray to consider for intersection. Defaults to infinity.

        Returns:
            Optional[PointQuadTree]: The closest node intersected by the ray, or None if no intersection occurs within maxLength.
        """
        # normalize direction vector
        direction = unitVec(direction)

        # helper function to check if ray intersects with a rectangle and get entry distance
        def rayIntersectsRect(point, direction, bounds, maxLength):
            minX, minY, maxX, maxY = bounds

            # parameter values for intersections with rectangle sides
            tXMin = (minX-point[0])/direction[0] if direction[0] != 0 else float('-inf' if point[0] < minX else 'inf')
            tXMax = (maxX-point[0])/direction[0] if direction[0] != 0 else float('-inf' if point[0] > maxX else 'inf')
            tYMin = (minY-point[1])/direction[1] if direction[1] != 0 else float('-inf' if point[1] < minY else 'inf')
            tYMax = (maxY-point[1])/direction[1] if direction[1] != 0 else float('-inf' if point[1] > maxY else 'inf')

            # ensure txmin <= txmax and tymin <= tymax
            if tXMin > tXMax:
                tXMin, tXMax = tXMax, tXMin
            if tYMin > tYMax:
                tYMin, tYMax = tYMax, tYMin
            
            # if ray doesnt intersect the rectangle
            if tXMin > tYMax or tYMin > tXMax:
                return False, float('inf')
            
            # parameters for closest and furthest intersections
            tMin = max(tXMin, tYMin)
            tMax = min(tXMax, tYMax)

            # if furthest intersection is behind ray or closest is beyond maxLength
            if tMax < 0 or tMin > maxLength:
                return False, float('inf')
            
            # if rect is around point
            if tMin < 0:
                return False, float('inf')
            
            return True, tMin
        
        # recursive function to find the closest rayintersecting leaf node
        def findClosestLeafNode(node, minDist, closestNode):
            # check if ray intersects this node
            intersects, dist = rayIntersectsRect(point, direction, node.bounds, maxLength)
            if not intersects or dist > minDist:
                return minDist, closestNode
            
            # if this is leaf node, check if its closer than current closest
            if not node.divided:
                if dist < minDist and len(child.points) > 0:
                    return dist, node
                return minDist, closestNode
            
            # if subdivided, check children
            # sort children by distance
            children = []
            for child in (node.northEast, node.northWest, node.southEast, node.southWest):
                intersects, dist = rayIntersectsRect(point, direction, child.bounds, maxLength)
                if intersects and dist < minDist and len(child.points) > 0:
                    children.append((child, dist))
            
            children.sort(key=lambda x: x[1])

            # check children in order of increasing distance
            for child, _ in children:
                minDist, closestNode = findClosestLeafNode(child, minDist, closestNode)

            return minDist, closestNode
        
        # start search from root
        minDist, closestNode = findClosestLeafNode(self, float('inf'), None)

        return closestNode
    
    def childrenPoints(self, pointsArray: List[int]) -> None:
        """
        Recursively collects all points from the leaf nodes and appends them to the provided array.

        This method traverses the entire tree, starting from the current node. If the node is subdivided,
        it recursively collects points from its children. If it is a leaf node, the points of that node are added
        to the provided `pointsArray`.

        Parameters:
            pointsArray (List[int]): The array where points from leaf nodes will be appended.

        Returns:
            None: This method modifies the provided `pointsArray` in place, but does not return anything.
    
        """
        if self.divided:
            self.southWest.childrenPoints(pointsArray)
            self.southEast.childrenPoints(pointsArray)
            self.northWest.childrenPoints(pointsArray)
            self.northEast.childrenPoints(pointsArray)
        else:
            # code here to run at each leaf node
            pointsArray.extend(self.points)
            return
    
# helper function to build quadtree
def buildPointQuadTree(triangulation: 'Triangulation') -> PointQuadTree:
    """
    Builds a PointQuadTree from a given triangulation object.

    This function initializes a PointQuadTree using the bounding box of the triangulation,
    and inserts points from the triangulation's coordinates into the quadtree. Each point is
    inserted by its index in the coordinate list.

    Parameters:
        triangulation (Triangulation): An object representing a triangulated mesh.
            It must have the following attributes:
            - coordinates: List of vertex coordinates in the mesh.
            - xMin: Minimum x-coordinate of the mesh.
            - yMin: Minimum y-coordinate of the mesh.
            - xMax: Maximum x-coordinate of the mesh.
            - yMax: Maximum y-coordinate of the mesh.

    Returns:
        PointQuadTree: A quadtree populated with the points from the triangulation.
    """
    # initialization
    qt = PointQuadTree(
        (triangulation.xMin-0.01, triangulation.yMin-0.01, triangulation.xMax+0.01, triangulation.yMax+0.01),
        triangulation.coordinates,
        capacity=8)
    # insert points
    for i in range(0, len(triangulation.coordinates), 2):
        qt.insert(i)
    
    return qt

# recursively iterates through all children
def children(node):
    # code here to run at each node
    # if -1 in node.triangles:
    #     print(node.triangles)
    if node.divided:
        children(node.southWest)
        children(node.southEast)
        children(node.northWest)
        children(node.northEast)
    else:
        # code here to run at each leaf node
        return

# def onQuadClick(qt, event):
#     # Check if the click was inside the axis
#     if event.inaxes:
#         # if right click
#         if event.button == 3:
#             # Get the x and y coordinates in data space
#             x, y = event.xdata, event.ydata
#             # triIdx = qt.searchPoint(x, y)
#             # drawTriangle(tri, triIdx//3)
#             leafNode = qt._findLeafNode(x, y)
#             if leafNode:
#             #     # print(leafNode.triangles)
#                 bounds = leafNode.bounds
#                 rect = Rectangle((bounds[0], bounds[1]), abs(bounds[0]-bounds[2]), abs(bounds[1]-bounds[3]), linewidth=1, edgecolor='r', facecolor='r', alpha=0.3)
#                 ax.add_patch(rect)
#                 for point in leafNode.points:
#                     px = qt.globalCoordinates[point]
#                     py = qt.globalCoordinates[point+1]
#                     print(px, py)
#                     ax.scatter(px, py)
#                 # px, py = qt.searchNearest(x, y)
#                 # ax.scatter(px, py)
#                 plt.draw()
