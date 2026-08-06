import math

from .triangulationSearch import doSegmentsIntersect, isOnSegment, \
    isCcwOrientation, adjacencyList, ballOfNode
from typing import TYPE_CHECKING, List, Optional
if TYPE_CHECKING:
    from .triangulation import Triangulation

EPSILON = math.pow(2,-52)
EDGE_STACK =[None] * 512

class Delaunator:

    def __init__(self,points):
        n = len(points)

        if (len(points) < 3):
            raise ValueError("Need at least 3 points")
        coords = [None] * n * 2

        for i in range(0,n):
            p = points[i]
            coords[2 * i] = (p[0])
            coords[2 * i+1] = (p[1])
        triangles = self.constructor(coords)
        # TODO: Look for how this algorithm can properly maintain
        # triangles in CCW arrangement.
        # properly turning triangle points to CCW
        for i in range(0, len(triangles), 3):
            temp = triangles[i+1]
            triangles[i+1] = triangles[i+2]
            triangles[i+2] = temp
        # reversedHull = reverseHull(pointCoords, hull)

    def constructor(self, coords):
        n = len(coords) >> 1

        self.coords = coords

        # arrays that will store the triangulation graph
        maxTriangles = max(2 * n - 5, 0)
        self._triangles = [None] * maxTriangles * 3
        self._halfedges = [None] * maxTriangles * 3

        # temporary arrays for tracking the edges of the advancing convex hull
        self.hashSize = math.ceil(math.sqrt(n))
        self.hullPrev = [None] * n # edge to prev edge
        self.hullNext = [None] * n # edge to next edge
        self.hullTri = [None] * n # edge to adjacent triangle
        self.hullHash = [-1] * self.hashSize # angular edge hash

        # temporary arrays for sorting points
        self._ids =  [None] * n
        self._dists = [None] * n
        triangles = self.update(coords)

        return triangles

    def update(self,coords):
        n = len(coords) >> 1

        # populate an array of point indices; calculate input data bbox
        minX = math.inf
        minY = math.inf
        maxX = -math.inf
        maxY = -math.inf

        for i in range(0,n):
            x = coords[2 * i]
            y = coords[2 * i + 1]
            if (x < minX): minX = x
            if (y < minY): minY = y
            if (x > maxX): maxX = x
            if (y > maxY): maxY = y
            self._ids[i] = i

        cx = (minX + maxX) / 2
        cy = (minY + maxY) / 2

        minDist = math.inf
        i0 = 0
        i1 = 0
        i2 = 0

        # pick a seed point close to the center
        for i in range(0,n):
            d = dist(cx, cy, coords[2 * i], coords[2 * i + 1])

            if (d < minDist):
                i0 = i
                minDist = d

        i0x = coords[2 * i0]
        i0y = coords[2 * i0 + 1]
        minDist = math.inf

        # find the point closest to the seed
        for i in range(0,n):
            if (i == i0): continue
            d = dist(i0x, i0y, coords[2 * i], coords[2 * i + 1])

            if (d < minDist and d > 0):
                i1 = i
                minDist = d

        i1x = coords[2 * i1]
        i1y = coords[2 * i1 + 1]

        minRadius = math.inf

        # find the third point which forms the smallest circumcircle with the first two
        for i in range(0,n):
            if (i == i0 or i == i1): continue
            r = circumradius(i0x, i0y, i1x, i1y, coords[2 * i], coords[2 * i + 1])

            if (r < minRadius):
                i2 = i
                minRadius = r

        i2x = coords[2 * i2]
        i2y = coords[2 * i2 + 1]

        if (minRadius == math.inf):
            # order collinear points by dx (or dy if all x are identical)
            # and return the list as a hull
            for i in range(0,n):
                self._dists[i] = (coords[2 * i] - coords[0]) or (coords[2 * i + 1] - coords[1])

            quicksort(self._ids, self._dists, 0, n - 1)
            hull =  [None] * n
            j = 0
            d0 = -math.inf

            for i in range(0,n):
                id = self._ids[i]

                if (self._dists[id] > d0):
                    hull[j] = id
                    j+=1
                    d0 = self._dists[id]

            self.hull = hull[0:j]
            self.triangles =  []
            self.halfedges =  []

        # swap the order of the seed points for counter-clockwise orientation
        if (orient(i0x, i0y, i1x, i1y, i2x, i2y)):
            i = i1
            x = i1x
            y = i1y
            i1 = i2
            i1x = i2x
            i1y = i2y
            i2 = i
            i2x = x
            i2y = y

        center = circumcenter(i0x, i0y, i1x, i1y, i2x, i2y)
        self._cx = center[0]
        self._cy = center[1]

        for i in range(0,n):
            self._dists[i] = dist(coords[2 * i], coords[2 * i + 1], center[0], center[1])

        # sort the points by distance from the seed triangle circumcenter
        quicksort(self._ids, self._dists, 0, n - 1)

        # set up the seed triangle as the starting hull
        self._hullStart = i0
        hullSize = 3

        self.hullNext[i0] = self.hullPrev[i2] = i1
        self.hullNext[i1] = self.hullPrev[i0] = i2
        self.hullNext[i2] = self.hullPrev[i1] = i0

        self.hullTri[i0] = 0
        self.hullTri[i1] = 1
        self.hullTri[i2] = 2

        self.hullHash[self._hashKey(i0x, i0y)] = i0
        self.hullHash[self._hashKey(i1x, i1y)] = i1
        self.hullHash[self._hashKey(i2x, i2y)] = i2

        self.trianglesLen = 0
        self._addTriangle(i0, i1, i2, -1, -1, -1)

        xp=0
        yp=0

        for k in range(0,len(self._ids)):
            i = self._ids[k]
            x = coords[2 * i]
            y = coords[2 * i + 1]

            # skip near-duplicate points
            if (k > 0 and abs(x - xp) <= EPSILON and abs(y - yp) <= EPSILON): continue

            xp = x
            yp = y

            # skip seed triangle points
            if (i == i0 or i == i1 or i == i2): continue

            # find a visible edge on the convex hull using edge hash
            start = 0
            key = self._hashKey(x, y)

            for j in range(0,self.hashSize):
                start = self.hullHash[(key + j) % self.hashSize]
                if (start != -1 and start != self.hullNext[start]): break

            start = self.hullPrev[start]
            e = start

            while True:
                q = self.hullNext[e]
                if orient(x, y, coords[2 * e], coords[2 * e + 1], coords[2 * q], coords[2 * q + 1]): break
                e = q

                if (e == start):
                    e = -1
                    break

            if (e == -1): continue # likely a near-duplicate point; skip it

            # add the first triangle from the point
            t = self._addTriangle(e, i, self.hullNext[e], -1, -1, self.hullTri[e])

            # recursively flip triangles from the point until they satisfy the Delaunay condition
            self.hullTri[i] = self._legalize(t + 2,coords)
            self.hullTri[e] = t # keep track of boundary triangles on the hull
            hullSize+=1

            # walk forward through the hull, adding more triangles and flipping recursively
            n = self.hullNext[e]

            while True:
                q = self.hullNext[n]
                if not (orient(x, y, coords[2 * n], coords[2 * n + 1], coords[2 * q], coords[2 * q + 1])): break
                t = self._addTriangle(n, i, q, self.hullTri[i], -1, self.hullTri[n])
                self.hullTri[i] = self._legalize(t + 2,coords)
                self.hullNext[n] = n # mark as removed
                hullSize-=1
                n = q

            # walk backward from the other side, adding more triangles and flipping
            if (e == start):
                while True:
                    q = self.hullPrev[e]
                    if not (orient(x, y, coords[2 * q], coords[2 * q + 1], coords[2 * e], coords[2 * e + 1])): break
                    t = self._addTriangle(q, i, e, -1, self.hullTri[e], self.hullTri[q])
                    self._legalize(t + 2,coords)
                    self.hullTri[q] = t
                    self.hullNext[e] = e # mark as removed
                    hullSize-=1
                    e = q

            # update the hull indices
            self._hullStart = self.hullPrev[i] = e
            self.hullNext[e] = self.hullPrev[n] = i
            self.hullNext[i] = n

            # save the two new edges in the hash table
            self.hullHash[self._hashKey(x, y)] = i
            self.hullHash[self._hashKey(coords[2 * e], coords[2 * e + 1])] = e

        self.hull = [None] * hullSize
        e = self._hullStart
        for i in range(0,hullSize):
            self.hull[i] = e
            e = self.hullNext[e]

        # trim typed triangle mesh arrays
        self.triangles = self._triangles[0:self.trianglesLen]
        self.halfedges = self._halfedges[0:self.trianglesLen]

        return self.triangles

    def _hashKey(self,x, y):
        return math.floor(pseudoAngle(x - self._cx, y - self._cy) * self.hashSize) % self.hashSize

    def _legalize(self,a,coords):
        i = 0
        ar = 0

        # recursion eliminated with a fixed-size stack
        while True:
            b = self._halfedges[a]
            """
              if the pair of triangles doesn't satisfy the Delaunay condition
              (p1 is inside the circumcircle of [p0, pl, pr]), flip them,
              then do the same check/flip recursively for the new pair of triangles
            """

            #         pl                    pl
            #        /||\                  /  \
            #     al/ || \bl            al/    \a
            #      /  ||  \              /      \
            #     /  a||b  \    flip    /___ar___\
            #   p0\   ||   /p1   =>   p0\---bl---/p1
            #      \  ||  /              \      /
            #     ar\ || /br             b\    /br
            #        \||/                  \  /
            #         pr                    pr

            a0 = a - a % 3
            ar = a0 + (a + 2) % 3

            if (b == -1): # convex hull edge
                if (i == 0): break
                i-=1
                a = EDGE_STACK[i]
                continue

            b0 = b - b % 3
            al = a0 + (a + 1) % 3
            bl = b0 + (b + 2) % 3

            p0 = self._triangles[ar]
            pr = self._triangles[a]
            pl = self._triangles[al]
            p1 = self._triangles[bl]

            illegal = inCircle(
                coords[2 * p0], coords[2 * p0 + 1],
                coords[2 * pr], coords[2 * pr + 1],
                coords[2 * pl], coords[2 * pl + 1],
                coords[2 * p1], coords[2 * p1 + 1])

            if (illegal):
                self._triangles[a] = p1
                self._triangles[b] = p0

                hbl = self._halfedges[bl]

                # edge swapped on the other side of the hull (rare); fix the halfedge reference
                if (hbl == -1):
                    e = self._hullStart
                    
                    while True:
                        if (self.hullTri[e] == bl):
                            self.hullTri[e] = a
                            break

                        e = self.hullPrev[e]
                        if (e == self._hullStart): break

                self._link(a, hbl)
                self._link(b, self._halfedges[ar])
                self._link(ar, bl)

                br = b0 + (b + 1) % 3

                # don't worry about hitting the cap: it can only happen on extremely degenerate input
                if (i < len(EDGE_STACK)):
                    EDGE_STACK[i] = br
                    i+=1

            else:
                if (i == 0): break
                i-=1
                a = EDGE_STACK[i]

        return ar

    def _link(self,a, b):
        self._halfedges[a] = b
        if (b != -1):
            self._halfedges[b] = a

    # add a new triangle given vertex indices and adjacent half-edge ids
    def _addTriangle(self,i0, i1, i2, a, b, c):
        t = self.trianglesLen

        self._triangles[t] = i0
        self._triangles[t + 1] = i1
        self._triangles[t + 2] = i2

        self._link(t, a)
        self._link(t + 1, b)
        self._link(t + 2, c)

        self.trianglesLen += 3

        return t

# monotonically increases with real angle, but doesn't need expensive trigonometry
def pseudoAngle(dx, dy):
    p = dx / (abs(dx) + abs(dy))

    if (dy > 0):
        return (3 - p) / 4 # [0..1]
    else:
        return (1 + p) / 4 # [0..1]

def dist(ax, ay, bx, by):
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy

# return 2d orientation sign if we're confident in it through J. Shewchuk's error bound check
def orientIfSure(px, py, rx, ry, qx, qy):
    l = (ry - py) * (qx - px)
    r = (rx - px) * (qy - py)

    if (abs(l - r) >= 3.3306690738754716e-16 * abs(l + r)):
        return l - r
    else:
        return 0

# a more robust orientation test that's stable in a given triangle (to fix robustness issues)
def orient(rx, ry, qx, qy, px, py):
    return (orientIfSure(px, py, rx, ry, qx, qy) or\
        orientIfSure(rx, ry, qx, qy, px, py) or\
        orientIfSure(qx, qy, px, py, rx, ry)) < 0

def inCircle(ax, ay, bx, by, cx, cy, px, py):
    dx = ax - px
    dy = ay - py
    ex = bx - px
    ey = by - py
    fx = cx - px
    fy = cy - py

    ap = dx * dx + dy * dy
    bp = ex * ex + ey * ey
    cp = fx * fx + fy * fy

    return dx * (ey * cp - bp * fy) -\
           dy * (ex * cp - bp * fx) +\
           ap * (ex * fy - ey * fx) < 0

def circumradius(ax, ay, bx, by, cx, cy):
    dx = bx - ax
    dy = by - ay
    ex = cx - ax
    ey = cy - ay

    bl = dx * dx + dy * dy
    cl = ex * ex + ey * ey
    try:
        d = 0.5/(dx * ey - dy * ex)
    except ZeroDivisionError:
        d = float('inf')

    x = (ey * bl - dy * cl) * d
    y = (dx * cl - ex * bl) * d

    return x*x + y*y

def circumcenter(ax, ay, bx, by, cx, cy):
    dx = bx - ax
    dy = by - ay
    ex = cx - ax
    ey = cy - ay

    bl = dx * dx + dy * dy
    cl = ex * ex + ey * ey
    try:
        d = 0.5/(dx * ey - dy * ex)
    except ZeroDivisionError:
        d = float('inf')

    x = ax + (ey * bl - dy * cl) * d
    y = ay + (dx * cl - ex * bl) * d

    return x, y

def quicksort(ids, dists, left, right):
    if (right - left <= 20):
        for i in range(left + 1,right+1):
            temp = ids[i]
            tempDist = dists[temp]
            j = i-1
            while (j >= left and dists[ids[j]] > tempDist):
                ids[j + 1] = ids[j]
                j-=1
            ids[j + 1] = temp;

    else:
        median = (left + right) >> 1
        i = left + 1
        j = right
        swap(ids, median, i)

        if (dists[ids[left]] > dists[ids[right]]):
            swap(ids, left, right)

        if (dists[ids[i]] > dists[ids[right]]):
            swap(ids, i, right)

        if (dists[ids[left]] > dists[ids[i]]):
            swap(ids, left, i)

        temp = ids[i]
        tempDist = dists[temp]

        while True:
            while True:
                i+=1
                if (dists[ids[i]] >= tempDist): break

            while True:
                j-=1
                if (dists[ids[j]] <= tempDist): break

            if (j < i): break
            swap(ids, i, j);

        ids[left + 1] = ids[j];
        ids[j] = temp;

        if (right - i + 1 >= j - left):
            quicksort(ids, dists, i, right)
            quicksort(ids, dists, left, j - 1)

        else:
            quicksort(ids, dists, left, j - 1)
            quicksort(ids, dists, i, right)

def swap(arr, i, j):
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp

def constrainedDelaunay(triangulation: 'Triangulation') -> None:
    """
    Constructs a constrained delaunay based on algorithm of Sloan 1993 paper.

    Parameters:
        triangulation (Triangulation): A triangulation object that contains point coordinates, 
                             triangle indices, adjacency information, and boundary definitions.

    Returns:
        None: The triangulation is modified in-place.
    """
    # local import because it created circular imports
    from .aflr import swapDiagonal

    coords = triangulation.coordinates
    triangles = triangulation.triangles
    adjacents = triangulation.adjacents
    
    # collecting boundary segment defined by p1 and p2 indexes in coords
    boundarySeg = triangulation.bdyPoints

    # for each edge
    for i in range(0, len(boundarySeg), 2):
        # segment definition
        segStart = boundarySeg[i]
        segEnd = boundarySeg[i+1]

        # finding all intersecting edges
        intersectingEdges = []
        
        # circling triangles at segStart
        # circlingTriIdx = trianglesHavingThisNode(triangles, segStart)
        circlingTriIdx = ballOfNode(triangulation, segStart)
        
        # if segment already exists, continue
        if doTrianglesHaveThisEdge(triangles, segStart, segEnd, circlingTriIdx): 
            continue
        
        # looping around circling triangles to find first edge intersecting segment
        for triIdx in circlingTriIdx:
            # finding opposite uIdx and vIdx
            if triangles[triIdx] == segStart:
                uIdx = triIdx+1 
                vIdx = triIdx+2 
            elif triangles[triIdx+1] == segStart:
                uIdx = triIdx+2 
                vIdx = triIdx 
            elif triangles[triIdx+2] == segStart:
                uIdx = triIdx 
                vIdx = triIdx+1 
            else:
                raise RuntimeError
            
            # checking if opposite edge intersects with segment
            if doSegmentsIntersect(coords[triangles[uIdx]], coords[triangles[uIdx]+1], coords[triangles[vIdx]], coords[triangles[vIdx]+1],\
                                   coords[segStart], coords[segStart+1], coords[segEnd], coords[segEnd+1]):
                # finding uAdjIdx and vAdjIdx
                adjTriIdx = adjacents[uIdx]
                vAdjIdx = findAdjV(triangulation, uIdx)

                # if adjacent is not -1
                if adjTriIdx != -1:
                    intersectingEdges.append(uIdx)
                    # drawSegment(triangulation, triIdx//3, uIdx%3)
                    break
        
        # marching forward to find all intersecting edges
        for j in range(len(triangles)):

            # errro in termination
            if j == len(triangles)-1: raise RuntimeError

            # checking if front node is segEnd
            frontNode = triangles[vAdjIdx - vAdjIdx%3 + (vAdjIdx-1)%3]
            if frontNode == segEnd: break

            # if segment intersects uf
            if doSegmentsIntersect(coords[triangles[uIdx]], coords[triangles[uIdx]+1], coords[frontNode], coords[frontNode+1], \
                                   coords[segStart], coords[segStart+1], coords[segEnd], coords[segEnd+1]):
                # shifting to next triangle
                uIdx = vAdjIdx - vAdjIdx%3 + (vAdjIdx+1)%3
                vAdjIdx = findAdjV(triangulation, uIdx)
                intersectingEdges.append(uIdx)
                # drawSegment(triangulation, (uIdx-uIdx%3)//3, uIdx%3)

            # if segment intersects fv
            else:
                # shifting to next triangle
                uIdx = vAdjIdx - vAdjIdx%3 + (vAdjIdx-1)%3
                vAdjIdx = findAdjV(triangulation, uIdx)
                intersectingEdges.append(uIdx)
                # drawSegment(triangulation, (uIdx-uIdx%3)//3, uIdx%3)
        
        # print(intersectingEdges)
        newEdges = []
        # removing intersecting edges
        for j in range(20000):

            if j == 20000-1: raise RuntimeError
            # if no edges intersecting
            if len(intersectingEdges) == 0: break

            # pop one edge out of array
            uIdx = intersectingEdges.pop()
            vAdjIdx = findAdjV(triangulation, uIdx)
            
            # check if quadrilateral is convex
            uNode = triangles[uIdx]
            vNode = triangles[vAdjIdx]
            aOpNode = triangles[uIdx - uIdx%3 + (uIdx-1)%3]
            bOpNode = triangles[vAdjIdx - vAdjIdx%3 + (vAdjIdx-1)%3]

            if not doSegmentsIntersect(coords[uNode], coords[uNode+1], coords[vNode], coords[vNode+1], \
                                   coords[aOpNode], coords[aOpNode+1], coords[bOpNode], coords[bOpNode+1]):
                # if not convex quadrilateral, unshift back the edge
                intersectingEdges.insert(0, uIdx)
                continue

            # if swapping results in flat triangle, unshift it back
            if isCcwOrientation(coords[uNode], coords[uNode+1], coords[bOpNode], coords[bOpNode+1], coords[aOpNode], coords[aOpNode]) == 0 or \
                isCcwOrientation(coords[vNode], coords[vNode+1], coords[bOpNode], coords[bOpNode+1], coords[aOpNode], coords[aOpNode]) == 0:
                intersectingEdges.insert(0, uIdx)
                continue

            # swap diagonal
            swapDiagonal(triangulation, uIdx, vAdjIdx-vAdjIdx%3 + (vAdjIdx-1)%3)
            
            # if new diagonal is not segment itself
            if (aOpNode != segStart and bOpNode != segEnd) and (aOpNode != segEnd and bOpNode != segStart):

                # finding new uIdx and vAdjIdx after swapping of diagonal
                uIdx = uIdx-uIdx%3 + (uIdx-1)%3
                vAdjIdx = findAdjV(triangulation, uIdx)

                # if new diagonal still intersects segment
                if doSegmentsIntersect(coords[aOpNode], coords[aOpNode+1], coords[bOpNode], coords[bOpNode+1], \
                                   coords[segStart], coords[segStart+1], coords[segEnd], coords[segEnd+1]):
                    # unshift new diagonal back to the list
                    intersectingEdges.insert(0, uIdx)
                else:
                    # push it to newEdges
                    newEdges.append(uIdx)

        # print(intersectingEdges)
        swapped = True
        # restoring delaunay property

        # while swapping happened
        for j in range(20000):

            # if not swapping, break
            if not swapped: break

            swapped = False

            # iterate over newEdges
            for k in range(len(newEdges)): 

                uIdx = newEdges[k]
                uNode = triangles[uIdx]
                vNode = triangles[uIdx-uIdx%3 + (uIdx+1)%3]

                # if new edge is segment itself, continue
                if (uNode == segStart and vNode == segEnd) or (uNode == segEnd and vNode == segStart): continue

                # if not delaunay, swap this edge
                if not isDelaunay(triangulation, uIdx):
                    vAdjIdx = findAdjV(triangulation, uIdx)
                    swapDiagonal(triangulation, uIdx, vAdjIdx-vAdjIdx%3 + (vAdjIdx-1)%3)
                    swapped = True

                    # finding new uIdx after swapping of diagonal
                    uIdx = uIdx-uIdx%3 + (uIdx-1)%3
                    # replace new diagonal into newEdges
                    newEdges[k] = uIdx

        # print(newEdges)

    # finding triangles outside of domain
    deleteFlags = [0]*(len(triangles)//3)
    
    # for all triangles
    for i in range(0, len(triangles), 3):

        # finding centroid of triangle
        cx = 1/3 * (coords[triangles[i]] + coords[triangles[i+1]] + coords[triangles[i+2]])
        cy = 1/3 * (coords[triangles[i]+1] + coords[triangles[i+1]+1] + coords[triangles[i+2]+1])

        # if centroid is not inside domain
        if not isPointInsidePoly(coords, boundarySeg, [cx, cy]):
            deleteFlags[i//3] =1

    deleteTris(triangulation, deleteFlags)

    # build attached triangles
    triangulation.adjPointTris = triangulation.attachTriangles()

# deletes triangles flagged by flags array in given triangulation
def deleteTris(triangulation: 'Triangulation', flags: list[int]) -> None:
    """
    Deletes triangles from the given triangulation based on the provided flags array.
    Each entry in the flags array corresponds to a triangle (group of 3 indices in the 
    triangulation array). If the flag is set (non-zero), the triangle is removed.

    The function also:
      - Updates the adjacency information to reflect the deleted triangles.
      - Rebuilds the adjacency list for the updated triangulation.
      - Removes coordinates that are no longer referenced by any triangle.

    Parameters:
        triangulation (Triangulation): The triangulation object containing mesh data.
        flags (list[int]): A list of flags where `1` indicates the triangle should be deleted 
                           and `0` means it should be retained.

    Returns:
        None: The function modifies the triangulation object in-place.
    """
    coords = triangulation.coordinates
    triangles = triangulation.triangles
    adjacents = triangulation.adjacents

    # deletion flags
    triFlags = [0]*(len(triangles))
    adjFlags = [0]*(len(adjacents))

    # for each flag
    for i in range(len(flags)):
        # if flag is off, continue
        if flags[i] == 0: continue

        triIdx = i*3
        # for all 3 edges of triangle
        for j in range(3):
            # setting flag to delete current triangle
            triFlags[triIdx+j] = 1

            # setting link to currently deleted triangle in adjacent triangles as -1
            if adjacents[triIdx+j] != -1:
                vAdjIdx = findAdjV(triangulation, triIdx+j)
                adjacents[vAdjIdx] = -1

            # setting flag to delete adjacent entries for currently deleted triangle
            adjFlags[triIdx+j] = 1

    newTriangles = []
    # deletion based on flags structure
    for i in range(len(triFlags)):
        if triFlags[i] != 1:
            newTriangles.append(triangles[i])

    # replacing tringulation triangles and adjacents
    triangulation.triangles = newTriangles

    # building adjacency list
    triangulation.adjacents = adjacencyList(newTriangles, len(triangulation.coordinates)//2)

    # deletion of coordinates which are not now part of new triangulation
    newCoords = []
    # for all coordinate indexes
    for i in range(0, len(coords), 2):
        # if current index not in triangles, continue
        if i not in triangulation.triangles: continue
        # append to newCoords
        newCoords.append(coords[i])
        newCoords.append(coords[i+1])
    
    # replacing coordinates
    triangulation.coordinates = newCoords

def isPointInsidePoly(coords: List[float], bdySegs: List[int], point: List[float]) -> bool:
    """
    Determines whether a given point is inside a polygon defined by boundary segments..
    
    Args:
        coords (List[float]): A flat list of x, y coordinate pairs (e.g., [x0, y0, x1, y1, ...]).
        bdySegs (List[int]): A list of indices defining boundary segments in pairs (e.g., [a, b, a, b, ...]),
                             where each pair refers to indices in the `coords` list.
        point (List[float]): The point (x, y) to test for inclusion in the polygon.

    Returns:
        bool: True if the point is inside the polygon or on its edge, False otherwise.

    Reference:
        https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.htmlclear
    
    TODO:
        merge it with isPointInsidePoly function in triangulationSearch.py
    """
    counter = 0
    N = len(bdySegs)
    for i in range(0, N, 2):
        p1 = [coords[bdySegs[i]], coords[bdySegs[i]+1]]
        p2 = [coords[bdySegs[i+1]], coords[bdySegs[i+1]+1]]
        
        if isOnSegment(p1[0], p1[1], point[0], point[1], p2[0], p2[1]):
            return True
        
        if point[1] > min(p1[1], p2[1]):
            if point[1] <= max(p1[1], p2[1]):
                if point[0] <= max(p1[0], p2[0]):
                    if p1[1] != p2[1]:
                        xInters = (point[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                        if p1[0] == p2[0] or point[0] <= xInters:
                            counter += 1

    return counter % 2 == 1

def isDelaunay(triangulation: 'Triangulation', uIdx: int) ->  bool:
    """
    Checks if the quadrilateral formed by two adjacent triangles in a triangulation
    satisfies the Delaunay (incircle) criterion.

    Parameters:
        triangulation (Triangulation): A triangulation object with `coordinates` and `triangles` attributes.
                                `coordinates` is a flat list of x, y coordinate pairs.
                                `triangles` is a list of indices into `coordinates` representing triangle vertices.
        uIdx (int): Index of the triangle vertex in the `triangles` list to start from.

    Returns:
        bool: True if the quadrilateral satisfies the Delaunay criterion, False otherwise.

    """
    coords = triangulation.coordinates
    triangles = triangulation.triangles

    # building quadrilateral structure
    uNode = triangles[uIdx]
    vNode = triangles[uIdx-uIdx%3 + (uIdx+1)%3]
    aOpNode = triangles[uIdx-uIdx%3 + (uIdx-1)%3]
    vAdjIdx = findAdjV(triangulation, uIdx)
    bOpNode = triangles[vAdjIdx-vAdjIdx%3 + (vAdjIdx-1)%3]

    # checking delaunay criterion
    if inCircle(coords[aOpNode], coords[aOpNode+1], coords[uNode], coords[uNode+1], \
                coords[vNode], coords[vNode+1], coords[bOpNode], coords[bOpNode+1]):
        return True
    else:
        return False
                
def findAdjV(triangulation: 'Triangulation', uIdx: int) -> int:
    """
    Finds the index of vertex `v` in the triangle adjacent to the triangle containing vertex at `uIdx`.

    Parameters:
        triangulation (Triangulation): A triangulation object with `triangles` and `adjacents` attributes.
                                `triangles` is a list of vertex indices in groups of three.
                                `adjacents` maps each triangle edge to its adjacent triangle index.
        uIdx (int): The index of the vertex `u` in the `triangles` list.

    Returns:
        int: The index in the `triangles` list of vertex `v` in the adjacent triangle.

    Raises:
        RuntimeError: If the adjacency data is inconsistent or the shared vertex cannot be found.
    """
    # finding adjacent triangle
    adjTriIdx = triangulation.adjacents[uIdx]

    if triangulation.triangles[adjTriIdx] == triangulation.triangles[uIdx]:
        vAdjIdx = adjTriIdx+2
    elif triangulation.triangles[adjTriIdx+1] == triangulation.triangles[uIdx]:
        vAdjIdx = adjTriIdx
    elif triangulation.triangles[adjTriIdx+2] == triangulation.triangles[uIdx]:
        vAdjIdx = adjTriIdx+1
    else:
        print(triangulation.triangles[uIdx], triangulation.triangles[adjTriIdx], triangulation.triangles[adjTriIdx+1], triangulation.triangles[adjTriIdx+2])
        # import matplotlib.pyplot as plt
        # from .triangulationDraw import drawTriangulation
        # drawTriangulation(triangulation)
        # plt.show()
        raise RuntimeError
    
    return vAdjIdx

def doTrianglesHaveThisEdge(triangles: list[int], p1: int, p2: int, circlingTriIdx: Optional[list[int]]=None):
    """
    Checks whether an edge defined by vertices `p1` and `p2` exists in the given triangle list.

    Parameters:
        triangles (list[int]): A flat list of vertex indices forming triangles in groups of three.
        p1 (int): The index of the first vertex of the edge.
        p2 (int): The index of the second vertex of the edge.
        circlingTriIdx (list[int] | None): Optional list of starting indices of specific triangles to search within.

    Returns:
        bool: True if the edge (p1, p2) exists in at least one triangle, False otherwise.
    """
    # if circling triangles list provided
    if circlingTriIdx:
        # for each circling triangle
        for idx in circlingTriIdx:
            triangleNodes = [triangles[idx], triangles[idx+1], triangles[idx+2]]
            # if p1p2 or p2p1 edge exists in triangle
            if p1 in triangleNodes and p2 in triangleNodes:
                return True
            
    # else, iterate on all triangles
    else:
        for idx in range(0, len(triangles), 3):
            triangleNodes = [triangles[idx], triangles[idx+1], triangles[idx+2]]
            # if p1p2 or p2p1 edge exists in triangle
            if p1 in triangleNodes and p2 in triangleNodes:
                return True


#################################################################

# import numpy as np

# def circumRadius(ax, ay, bx, by, cx, cy):
#     bx -= ax
#     by -= ay
#     cx -= ax
#     cy -= ay

#     bl = bx*bx + by*by
#     cl = cx*cx + cy*cy

#     try:
#         d = 0.5/(bx*cy - by*cx)
#     except ZeroDivisionError:
#         return np.inf

#     x = (cy*bl - by*cl) * d
#     y = (bx*cl - cx*bl) * d

#     return x*x + y*y

# # @TODO improve with robust areaOrient predictate
# # returns positive for CCW points
# def areaOrient(ax, ay, bx, by, cx, cy):
#     return (by-ay)*(cx-bx) - (bx-ax)*(cy-by)

# def circumCenter(ax, ay, bx, by, cx, cy):
#     bx -= ax
#     by -= ay
#     cx -= ax
#     cy -= ay

#     bl = bx*bx + by*by
#     cl = cx*cx + cy*cy

#     d = 0.5/(bx*cy - by*cx)

#     x = (cy*bl - by*cl) * d
#     y = (bx*cl - cx*bl) * d

#     return [ax+x, ay+y]

# def insertNode(coords, index, prev=None):
#     node = {
#         'i': index,
#         'x': coords[index],
#         'y': coords[index+1],
#         't': 0,
#         'prev': None,
#         'next': None
#     }

#     if prev == None:
#         node['prev'] = node
#         node['next'] = node
#     else:
#         node['next'] = prev['next']
#         node['prev'] = prev
#         prev['next']['prev'] = node
#         prev['next'] = node

#     return node

# def removeNode(node):
#     node['prev']['next'] = node['next']
#     node['next']['prev'] = node['prev']
#     return node['prev']

# def insertTriangle(triangles, i, nextHull):
#     t = len(triangles)
#     triangles.append(nextHull['i'])
#     triangles.append(i)
#     triangles.append(nextHull['next']['i'])
#     return t

# def linkAdjacent(a, b, adjacent):
#     if (a >= len(adjacent)): 
#         adjacent.append(b)
#     else:
#         adjacent[a] = b
#     if (b != -1): adjacent[b] = a

# # @TODO improve with robust inCircle predictate
# # returns true if point d is inside circle by a,b,c
# def inCircle(ax, ay, bx, by, cx, cy, dx, dy):
#     ax -= dx
#     ay -= dy
#     bx -= dx
#     by -= dy
#     cx -= dx
#     cy -= dy

#     p = ax * (by*(cx*cx+cy*cy) - cy*(bx*bx+by*by))
#     q = ay * (bx*(cx*cx+cy*cy) - cx*(bx*bx+by*by))
#     r = (ax*ax + ay*ay) * (bx*cy - by*cx)

#     det = p-q+r
#     return det < 0

# def legalize(a, adjacent, triangles, pointCoords):
#     b = adjacent[a]

#     a0 = a-a%3
#     b0 = b-b%3

#     al = a0 + (a+1)%3
#     ar = a0 + (a+2)%3
#     br = b0 + (b+1)%3
#     bl = b0 + (b+2)%3

#     p0 = triangles[ar]
#     pr = triangles[a]
#     pl = triangles[al]
#     p = triangles[bl]

#     illegal = inCircle(
#             pointCoords[p0], pointCoords[p0+1],
#             pointCoords[pr], pointCoords[pr+1],
#             pointCoords[pl], pointCoords[pl+1],
#             pointCoords[p], pointCoords[p+1]
#         )
    
#     if illegal:
#         triangles[a] = p
#         triangles[b] = p0

#         linkAdjacent(a, adjacent[bl], adjacent)
#         linkAdjacent(b, adjacent[ar], adjacent)
#         linkAdjacent(ar, bl, adjacent)

#         legalize(a, adjacent, triangles, pointCoords)
#         return legalize(br, adjacent, triangles, pointCoords)
    
#     return ar

# def reverseHull(coords, hull):
#     hullStart = hull
#     revHull = insertNode(coords, hull['i'])

#     hull = hull['prev']
#     while (hull != hullStart):
#         revHull = insertNode(coords, hull['i'], revHull)
#         hull = hull['prev']

#     return revHull

# # takes point coordinates in [x, y, x, y, ...] form
# def delaunate(pointCoords):
#     pointCoords = (np.array(pointCoords, dtype=np.float64))
#     # print(pointCoords)
#     # # array storing ids of pointCoords
#     # ids = np.arange(pointCoords.size, dtype=np.uint64)

#     # finding centroid of given vertices
#     cx = np.average(pointCoords[0::2])
#     cy = np.average(pointCoords[1::2])
#     # print(cx, cy)

#     # finding p0, closest to the centroid (cx, cy)
#     p0 = 0
#     minD = np.inf
#     for idx in range(0, pointCoords.size, 2):
#         distSquared = (pointCoords[idx]-cx)**2 + (pointCoords[idx+1]-cy)**2
#         if (distSquared < minD):
#             p0 = idx
#             minD = distSquared
#     # print(p0)

#     # finding p1, closest to the p0
#     p1 = 0
#     minD = np.inf
#     for idx in range(0, pointCoords.size, 2):
#         if idx == p0: continue
#         distSquared = (pointCoords[idx]-pointCoords[p0])**2 + (pointCoords[idx+1]-pointCoords[p0+1])**2
#         if (distSquared < minD and distSquared > 0): 
#             p1 = idx
#             minD = distSquared
#     # print(p1)

#     # finding p2, which miminizes circumcircle radius with p0 and p1
#     p2 = 0
#     minRad = np.inf
#     for idx in range(0, pointCoords.size, 2):
#         if idx == p0 or idx == p1: continue
#         rad = circumRadius(pointCoords[p0], pointCoords[p0+1],
#             pointCoords[p1], pointCoords[p1+1],
#             pointCoords[idx], pointCoords[idx+1])

#         if rad < minRad:
#             p2 = idx
#             minRad = rad
#     # print(p2)
    
#     # @TODO
#     # check for completely collinear points
#     if (minRad == np.inf): raise Exception('(Collinear points) something\'s fishy!')

#     # Check for CCW triangle, if not, flip points p1 and p2
#     if areaOrient(
#         pointCoords[p0], pointCoords[p0+1],
#         pointCoords[p1], pointCoords[p1+1],
#         pointCoords[p2], pointCoords[p2+1]
#     ) < 0:
#         temp = p1
#         p1 = p2
#         p2 = temp
    
#     # finding circumcenter of seed triangle
#     center = circumCenter(
#         pointCoords[p0], pointCoords[p0+1],
#         pointCoords[p1], pointCoords[p1+1],
#         pointCoords[p2], pointCoords[p2+1]
#     )

#     # sorting point ids in ascending for distance from circumcenter
#     dists = (pointCoords[0::2])**2 + (pointCoords[1::2])**2
#     sortedIds = dists.argsort()*2

#     # initialize convex hull as seed triangle
#     hull = insertNode(pointCoords, p0)
#     hull['t'] = 0
#     hull = insertNode(pointCoords, p1, hull)
#     hull['t'] = 1
#     hull = insertNode(pointCoords, p2, hull)
#     hull['t'] = 2
#     # print(hull)

#     # initialize triangles and adjacent arrays
#     # -1 in adjacent refers to outer hull edge with no adjacent edge
#     # triangles = np.array([p0, p1, p2], dtype=np.uint64)
#     # adjacent = np.array([-1, -1, -1], dtype=np.int64)
#     triangles = [p0, p1, p2]
#     adjacent = [-1, -1, -1]
#     # print(triangles, adjacent)

#     xp = yp = None
#     for k in range(sortedIds.size):
#         i = sortedIds[k]
#         x = pointCoords[i]
#         y = pointCoords[i+1]
#         if (i==p0 or i==p1 or i==p2): continue # skip seed triangle points
#         if (x==xp and y==yp): continue # skip duplicate points

#         # save this point as last point
#         xp = x
#         yp = y

#         # check if new point on same/opposite side of hull start,
#         # accordingly do forward-backward pass for adding triangles or
#         # do one forward pass adding all triangles in one go
#         hullStart = hull
#         while (areaOrient(x, y, hullStart['x'], hullStart['y'], hullStart['next']['x'], hullStart['next']['y']) >= 0):
#             hullStart = hullStart['next']

#             # # @TODO: what this means?
#             # if (hullStart == hull): raise Exception('fishy2')
        
#         backPass = hullStart == hull

#         # add first triangle and adjacent edges
#         t = insertTriangle(triangles, i, hullStart)
#         adjacent.append(-1)
#         adjacent.append(-1)
#         linkAdjacent(t+2, hullStart['t'], adjacent) # link shared edges of new triangle and triangle at hull

#         # advance hull over added triangle
#         hullStart['t'] = t
#         hullStart = insertNode(pointCoords, i, hullStart)
#         hullStart['t'] = legalize(t+2, adjacent, triangles, pointCoords)

#         # adding triangles in forward pass and legalizing them
#         hullNext = hullStart['next']
#         while(areaOrient(x, y, hullNext['x'], hullNext['y'], hullNext['next']['x'], hullNext['next']['y']) < 0):
#             t = insertTriangle(triangles, i, hullNext)

#             linkAdjacent(t, hullNext['prev']['t'], adjacent)
#             adjacent.append(-1)
#             linkAdjacent(t+2, hullNext['t'], adjacent)

#             hullNext['prev']['t'] = legalize(t+2, adjacent, triangles, pointCoords)

#             hull = removeNode(hullNext)
#             hullNext = hullNext['next']

#         # do backpass if required for adding triangles
#         if not backPass: continue
#         hullPrev = hullStart['prev']
#         while(areaOrient(x, y, hullPrev['prev']['x'], hullPrev['prev']['y'], hullPrev['x'], hullPrev['y']) < 0):
#             t = insertTriangle(triangles, i, hullPrev['prev'])

#             adjacent.append(-1)
#             linkAdjacent(t+1, hullPrev['t'], adjacent)
#             linkAdjacent(t+2, hullPrev['prev']['t'], adjacent)

#             legalize(t+2, adjacent, triangles, pointCoords)

#             hullPrev['prev']['t'] = t
#             hull = removeNode(hullPrev)
#             hullPrev = hullPrev['prev']

#     # TODO: Look for how this algorithm can properly maintain
#     # triangles in CCW arrangement.
#     # properly turning triangle points to CCW
#     for i in range(0, len(triangles), 3):
#         temp = triangles[i+1]
#         triangles[i+1] = triangles[i+2]
#         triangles[i+2] = temp
#     reversedHull = reverseHull(pointCoords, hull)

#     return{
#         'coordinates': pointCoords,
#         'triangles': triangles,
#         'adjacents': adjacent,
#         'hull': reversedHull
#     }

