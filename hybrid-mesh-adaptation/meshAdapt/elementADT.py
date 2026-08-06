from typing import TYPE_CHECKING, List, Optional, Tuple
if TYPE_CHECKING:
    from .triangulation import Triangulation

from .triangulationSearch import isPointInsideTriangle, isPointOnTriangle

# Number of dimensions of the ADT search space. Each triangle is represented
# by its axis aligned bounding box (xmin, ymin, xmax, ymax), i.e. a point in
# a 4D space (2*d for d=2). The splitting dimension at each tree level cycles
# through 0, 1, 2, 3, 0, 1, ...
NDIMS = 4

# Tolerance added to bounding-box containment checks. Points produced by
# curve projection or barycentric interpolation can be off by a tiny
# floating point amount, which would otherwise place them just outside
# every adjacent triangle's bounding box and make searchElement miss a
# triangle that isPointOnTriangle would accept. Matches the geometric
# tolerance used elsewhere (triangulation.epsilon).
BBOX_EPSILON = 1e-10


class ElementADT:
    """
    A class representing an Alternating Digital Tree (ADT) used to efficiently
    determine which mesh element (triangle) contains a query point.

    Every triangle is stored using its axis aligned bounding box
    (xmin, ymin, xmax, ymax), which is treated as a point in a 4D space. The
    tree is a binary tree where the splitting dimension alternates with the
    depth of the node (dim = level % 4), cycling through xmin, ymin, xmax and
    ymax in turn.

    To find the element containing a query point (x, y), the tree is searched
    for every stored bounding box that satisfies:
        xmin <= x <= xmax and ymin <= y <= ymax
    Each node also stores the bounding region (in the 4D ADT space) of its
    whole subtree, which is used to prune branches that cannot contain a
    matching bounding box. The resulting candidate elements are then checked
    with an exact point-in-triangle test.

    Attributes:
        boxPoint (Tuple[float, float, float, float]): The (xmin, ymin, xmax, ymax)
            bounding box of the element stored at this node.
        elementIdx (int): The index (into triangulation.triangles) of the
            element stored at this node.
        level (int): The depth of this node in the tree, used to determine the
            splitting dimension (level % 4).
        left (Optional[ElementADT]): Child node containing elements whose
            bounding box coordinate in the splitting dimension is smaller.
        right (Optional[ElementADT]): Child node containing elements whose
            bounding box coordinate in the splitting dimension is greater or
            equal.
        regionMin (List[float]): Component-wise minimum, over all elements in
            this subtree, of (xmin, ymin, xmax, ymax).
        regionMax (List[float]): Component-wise maximum, over all elements in
            this subtree, of (xmin, ymin, xmax, ymax).
        parent (Optional[ElementADT]): This node's parent, or None for the
            root. Used by `update` to walk back up to the root after a
            node's box changes.
        nodeMap (Dict[int, ElementADT]): Shared across every node in the
            tree, maps an element's index (into triangulation.triangles) to
            the node storing it, for O(1) lookup in `update`.

    Reference:
        Bonet, J. and Peraire, J., "An Alternating Digital Tree (ADT) algorithm
        for 3D geometric searching and intersection problems", International
        Journal for Numerical Methods in Engineering, 1991.
    """
    def __init__(
            self,
            boxPoint: Tuple[float, float, float, float],
            elementIdx: int,
            level: Optional[int]=0,
            parent: Optional['ElementADT']=None,
            nodeMap: Optional[dict]=None) -> None:
        self.boxPoint = boxPoint
        self.elementIdx = elementIdx
        self.level = level
        self.parent = parent
        self.left: Optional['ElementADT'] = None
        self.right: Optional['ElementADT'] = None

        # bounding region (in 4D ADT space) of this node's subtree.
        # initialized to this node's own point, expanded on every insertion.
        self.regionMin = list(boxPoint)
        self.regionMax = list(boxPoint)

        # shared by every node in the tree; register this node for update()
        self.nodeMap = nodeMap if nodeMap is not None else {}
        self.nodeMap[elementIdx] = self

    def insert(self, boxPoint: Tuple[float, float, float, float], elementIdx: int) -> None:
        """
        Inserts an element's bounding box into the ADT.

        Every node's subtree region along the path is expanded to include the
        new bounding box, then the box is placed in the left or right child
        depending on its coordinate in that node's splitting dimension
        (level % 4). Walked iteratively (rather than recursing node-to-node)
        since the alternating-axis BST has no balancing step, so its depth
        can grow with insertion order/size and a recursive walk can exceed
        Python's recursion limit on large meshes.

        Parameters:
            boxPoint (Tuple[float, float, float, float]): The (xmin, ymin, xmax, ymax)
                bounding box of the element to insert.
            elementIdx (int): The index (into triangulation.triangles) of the
                element to insert.

        Returns:
            None
        """
        node = self

        while True:
            # expand this node's subtree bounding region to include the new point
            for d in range(NDIMS):
                if boxPoint[d] < node.regionMin[d]:
                    node.regionMin[d] = boxPoint[d]
                if boxPoint[d] > node.regionMax[d]:
                    node.regionMax[d] = boxPoint[d]

            # alternate the splitting dimension with tree depth
            dim = node.level % NDIMS

            if boxPoint[dim] < node.boxPoint[dim]:
                if node.left is None:
                    node.left = ElementADT(boxPoint, elementIdx, node.level + 1, parent=node, nodeMap=node.nodeMap)
                    return
                node = node.left
            else:
                if node.right is None:
                    node.right = ElementADT(boxPoint, elementIdx, node.level + 1, parent=node, nodeMap=node.nodeMap)
                    return
                node = node.right

    def update(self, elementIdx: int, newBoxPoint: Tuple[float, float, float, float]) -> None:
        """
        Updates the bounding box of an already-inserted element, e.g. after
        its underlying triangle's vertices changed due to a subdivision or
        diagonal swap.

        The node's own `boxPoint` is replaced exactly. The `regionMin`/
        `regionMax` of the node and every ancestor up to the root are then
        expanded (never shrunk) to include the new box. Regions are only
        ever expanded so that `_regionMayContainPoint` pruning stays safe
        (it never excludes a subtree that could contain a match), even
        though the bounds may become slightly looser over time.

        This is an O(depth) operation, unlike rebuilding the whole tree.

        Parameters:
            elementIdx (int): The index (into triangulation.triangles) of the
                element whose bounding box changed.
            newBoxPoint (Tuple[float, float, float, float]): The element's
                updated (xmin, ymin, xmax, ymax) bounding box.

        Returns:
            None
        """
        node = self.nodeMap[elementIdx]
        node.boxPoint = newBoxPoint

        while node is not None:
            for d in range(NDIMS):
                if newBoxPoint[d] < node.regionMin[d]:
                    node.regionMin[d] = newBoxPoint[d]
                if newBoxPoint[d] > node.regionMax[d]:
                    node.regionMax[d] = newBoxPoint[d]
            node = node.parent

    def _regionMayContainPoint(self, x: float, y: float) -> bool:
        """
        Checks whether this node's subtree can possibly hold an element whose
        bounding box contains the point (x, y).

        A bounding box (xmin, ymin, xmax, ymax) contains (x, y) when
        xmin <= x, ymin <= y, xmax >= x and ymax >= y. This method checks
        whether the subtree's region allows for any of those four conditions
        to hold; if any of them is impossible for every element in the
        subtree, the whole subtree can be skipped during search.

        Parameters:
            x (float): The x-coordinate of the query point.
            y (float): The y-coordinate of the query point.

        Returns:
            bool: False if no element in this subtree can contain (x, y),
                True otherwise.
        """
        if self.regionMin[0] - BBOX_EPSILON > x:  # smallest xmin in subtree already > x
            return False
        if self.regionMin[1] - BBOX_EPSILON > y:  # smallest ymin in subtree already > y
            return False
        if self.regionMax[2] + BBOX_EPSILON < x:  # largest xmax in subtree already < x
            return False
        if self.regionMax[3] + BBOX_EPSILON < y:  # largest ymax in subtree already < y
            return False

        return True

    def _boxContainsPoint(self, x: float, y: float) -> bool:
        """
        Checks if this node's own bounding box contains the point (x, y).
        """
        xmin, ymin, xmax, ymax = self.boxPoint

        return (xmin - BBOX_EPSILON) <= x <= (xmax + BBOX_EPSILON) and (ymin - BBOX_EPSILON) <= y <= (ymax + BBOX_EPSILON)

    def _collectCandidates(self, x: float, y: float, candidates: List[Tuple[int, Tuple[float, float, float, float]]]) -> None:
        """
        Recursively collects every element whose bounding box contains the
        point (x, y), pruning subtrees whose region cannot contain a match.

        Parameters:
            x (float): The x-coordinate of the query point.
            y (float): The y-coordinate of the query point.
            candidates (List[Tuple[int, Tuple[float, float, float, float]]]):
                List that (elementIdx, boxPoint) pairs are appended to.

        Returns:
            None
        """
        if not self._regionMayContainPoint(x, y):
            return

        if self._boxContainsPoint(x, y):
            candidates.append((self.elementIdx, self.boxPoint))

        if self.left:
            self.left._collectCandidates(x, y, candidates)
        if self.right:
            self.right._collectCandidates(x, y, candidates)

    def searchElement(self, x: float, y: float, triangulation: 'Triangulation') -> int:
        """
        Finds the index of the triangle in `triangulation` that contains the
        point (x, y).

        The ADT is first queried for every element whose bounding box contains
        (x, y). These candidates are checked smallest-bounding-box first with
        an exact point-in-triangle test. If none of the candidates strictly
        contains the point (e.g. the point lies exactly on a shared edge or
        vertex), the candidates are checked again allowing the point to lie on
        the triangle's boundary.

        Parameters:
            x (float): The x-coordinate of the query point.
            y (float): The y-coordinate of the query point.
            triangulation (Triangulation): The triangulation to search, used
                for its `coordinates` and `triangles` arrays.

        Returns:
            int: The index of the triangle containing (x, y), or -1 if no
                such triangle was found.
        """
        candidates: List[Tuple[int, Tuple[float, float, float, float]]] = []
        self._collectCandidates(x, y, candidates)

        # try the tightest-fitting bounding boxes first
        candidates.sort(key=lambda c: _bboxArea(c[1]))

        coords = triangulation.coordinates
        triangles = triangulation.triangles

        # exact containment test
        for elementIdx, _ in candidates:
            if isPointInsideTriangle(coords, triangles, elementIdx, [x, y]):
                return elementIdx

        # fall back to boundary tolerance for points on shared edges/vertices
        for elementIdx, _ in candidates:
            if isPointOnTriangle(coords, triangles, elementIdx, [x, y]):
                return elementIdx

        return -1


def _bboxArea(box: Tuple[float, float, float, float]) -> float:
    """
    Returns the area of an axis aligned bounding box (xmin, ymin, xmax, ymax).
    """
    xmin, ymin, xmax, ymax = box

    return (xmax - xmin) * (ymax - ymin)


def elementBoundingBox(triangulation: 'Triangulation', triIdx: int) -> Tuple[float, float, float, float]:
    """
    Computes the axis aligned bounding box (xmin, ymin, xmax, ymax) of the
    triangle starting at index `triIdx` in `triangulation.triangles`.
    """
    coords = triangulation.coordinates
    triangles = triangulation.triangles

    xs = (coords[triangles[triIdx]], coords[triangles[triIdx + 1]], coords[triangles[triIdx + 2]])
    ys = (coords[triangles[triIdx] + 1], coords[triangles[triIdx + 1] + 1], coords[triangles[triIdx + 2] + 1])

    return (min(xs), min(ys), max(xs), max(ys))


# helper function to build the element ADT
def buildElementADT(triangulation: 'Triangulation') -> ElementADT:
    """
    Builds an ElementADT from a given triangulation object.

    Each triangle's axis aligned bounding box is computed from its three
    vertices and inserted into the ADT, keyed by the triangle's starting
    index in `triangulation.triangles`.

    Parameters:
        triangulation (Triangulation): An object representing a triangulated
            mesh. It must have the following attributes:
            - coordinates: List of vertex coordinates in the mesh.
            - triangles: Flattened list of triangles, each defined by three
              point indices into `coordinates`.

    Returns:
        ElementADT: The root node of the ADT populated with the bounding boxes
            of every triangle in the triangulation.
    """
    triangles = triangulation.triangles

    root = None

    for triIdx in range(0, len(triangles), 3):
        boxPoint = elementBoundingBox(triangulation, triIdx)

        if root is None:
            root = ElementADT(boxPoint, triIdx, level=0)
        else:
            root.insert(boxPoint, triIdx)

    return root
