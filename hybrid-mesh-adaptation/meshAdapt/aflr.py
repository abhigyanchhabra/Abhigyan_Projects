import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from typing import List, Union, Optional, Callable

from .metric import lengthInMetricMetric, lengthInMetric, eigen, unitVec, \
    metricUnitPoint, angleInMetric, angle, logMetric, boundaryMesh, dotProduct
from .triangulationSearch import boundaryProject, closestBoundaryPoint, \
    doSegmentsIntersect, adjacencyList, searchTriangle, ballOfNode, findNearestTriangle
from .triangulation import Triangulation
from .triangulationDraw import drawTriangulation, drawTriangle
from .pointQuadTree import buildPointQuadTree
from .elementADT import elementBoundingBox
from .delaunay import inCircle

def metricSizingFlags(triangulation: Triangulation) -> List[int]:
    """
    Gives flags for given triangulation satisfying metric sizing criteria

    Parameters:
        triangulation (Triangulation): An object representing a triangulated mesh. 
            It must have the following attributes:
            - triangles: List of triangle vertex indices (flattened).
            - coordinates: List of vertex coordinates.
            - metricMesh: List of metric tensor values per vertex.

    Returns:
        List[int]: A list of flags (1 for active, 0 for inactive) for each triangle.

    Notes:
        - Edge is too long if length ≥ sqrt(2) * `cUpper`
        - Edge is too short if length ≤ 1 / sqrt(2) * `cLower`

    """
    # length criterion multiplier
    cUpper = 1.3
    cLower = 1.0

    # adding flags, 1/0 is active/off initially all are active
    flags = [1]*int(len(triangulation.triangles)/3)

    coords = triangulation.coordinates
    metricMesh = triangulation.metricMesh
    # turn off flag for triangles which satisfy metric sizing criterion
    for i in range(0, len(triangulation.triangles), 3):
        counter = 0
        lengths = []
        # if bl triangle, consider it off
        if triangulation.blFlags and i//3 < len(triangulation.blFlags) and triangulation.blFlags[i//3] == 0:
            flags[i//3] = 0
            continue

        for j in range(3):
            segStart = [coords[triangulation.triangles[i+j]], coords[triangulation.triangles[i+j]+1]]
            segEnd = [coords[triangulation.triangles[i+(j+1)%3]], coords[triangulation.triangles[i+(j+1)%3]+1]]
            segStartMetric = [metricMesh[int(triangulation.triangles[i+j]*1.5)], metricMesh[int(triangulation.triangles[i+j]*1.5)+1], metricMesh[int(triangulation.triangles[i+j]*1.5)+2]]
            segEndMetric = [metricMesh[int(triangulation.triangles[i+(j+1)%3]*1.5)], metricMesh[int(triangulation.triangles[i+(j+1)%3]*1.5)+1], metricMesh[int(triangulation.triangles[i+(j+1)%3]*1.5)+2]]
            length = lengthInMetricMetric(segStartMetric, segEndMetric, segStart, segEnd)
            if length >= (2**0.5)*cUpper or length <= (1/2**0.5)*cLower: 
                counter+=1
        
        if counter == 0:
            flags[int(i/3)] = 0

        if triangulation.blFlags:
            if i//3 < len(triangulation.blFlags) and triangulation.blFlags[i//3] == 0:
                flags[i//3] = 0
    
    return flags

def findCandidatePoints(
        triangulation: Triangulation, 
        segStartIdx: int,
        segEndIdx: int,
        metric: Callable[[float, float], List[float]]) -> List[List[float]]:
    
    """
    Finds candidate points along a given edge of a triangulation based on the metric at the edge endpoints.

    Parameters:
        triangulation (Triangulation): An object representing a triangulated mesh.
            It must have the following attributes:
            - coordinates: List of vertex coordinates in the mesh.
            - metricMesh: List of metric tensor values at each vertex.
            - bdyPoints: Boundary points of the triangulation.
        segStartIdx (int): Index of the first endpoint of the segment in the triangulation's coordinates.
        segEndIdx (int): Index of the second endpoint of the segment in the triangulation's coordinates.
        metric (Callable[[float, float], List[float]]): A function that computes the metric tensor at a given point.
            It takes coordinates (x, y) and returns a list of metric tensor components.

    Returns:
        List[List[float]]: A list containing the following four lists:
            - proposedPoint1: A list representing the coordinates of the first proposed point (or -1 if invalid).
            - proposedPoint2: A list representing the coordinates of the second proposed point (or -1 if invalid).
            - proposedPoint1Metric: A list representing the metric tensor at the first proposed point (or -1 if invalid).
            - proposedPoint2Metric: A list representing the metric tensor at the second proposed point (or -1 if invalid).
    """

    coords = triangulation.coordinates
    metricMesh = triangulation.metricMesh

    # min and max anisotropic ratios
    rmin = 1
    rmax = 100
    
    # finding coordinates of edge endpoints
    xp1 = coords[segStartIdx]
    yp1 = coords[segStartIdx+1]
    xp2 = coords[segEndIdx]
    yp2 = coords[segEndIdx+1]

    # edge normal
    edgeNormal = [-(yp2-yp1), xp2-xp1]

    # fetching metric at p1 and p2
    metricP1 = [metricMesh[int(segStartIdx*1.5)], metricMesh[int(segStartIdx*1.5)+1], metricMesh[int(segStartIdx*1.5)+2]]
    metricP2 = [metricMesh[int(segEndIdx*1.5)], metricMesh[int(segEndIdx*1.5)+1], metricMesh[int(segEndIdx*1.5)+2]]

    # eigenvectors at p1 and p2
    eigenObjectP1 = eigen(metricP1)
    eigenObjectP2 = eigen(metricP2)

    # point 1
    # finding point q1/2 with eigenvector alined to edgenormal at p1/2
    vp1 = _mostAlignedEigenVector(eigenObjectP1['vec1'], eigenObjectP1['vec2'], edgeNormal)
    tp1 = (eigenObjectP1['lambda1'] if vp1 == eigenObjectP1['vec1'] else eigenObjectP1['lambda2'])**(-0.5)
    xq1, yq1 = [xp1 + vp1[0]*tp1, yp1 + vp1[1]*tp1]


    # if q1 outside boundary leave point proposition
    if searchTriangle(xq1, yq1, triangulation) == -1:
        proposedPoint1 = -1
        proposedPoint1Metric = -1
    else:
        # finding metric at q1
        metricQ1 = metric(xq1, yq1)

        # finding eigvec at q1
        eigenObjectQ1 = eigen(metricQ1)

        # finding aligned eigenvec at q1 with edgenormal
        vq1 = _mostAlignedEigenVector(eigenObjectQ1['vec1'], eigenObjectQ1['vec2'], edgeNormal)

        # average both eigenvectors vp1 and vq1
        vpq1 = [0.5*(vp1[0] + vq1[0]), 0.5*(vp1[1] + vq1[1])]

        # metric anisotropic ratio r1
        r1 = (max(eigenObjectP1['lambda1'], eigenObjectP1['lambda2'])/min(eigenObjectP1['lambda1'], eigenObjectP1['lambda2']))**0.5

        beta1 = 1 - max(0, min((r1-rmin)/(rmax-rmin), 1))
        edgeNormalUnit = unitVec(edgeNormal)
        vopt1 = [beta1*edgeNormalUnit[0] + (1-beta1)*vpq1[0], beta1*edgeNormalUnit[1] + (1-beta1)*vpq1[1]]
        # vopt1 = [beta1*edgeNormalUnit[0] + (1-beta1)*vp1[0], beta1*edgeNormalUnit[1] + (1-beta1)*vp1[1]] # vopt = beta*normal + (1-beta)*vp

        vopt1End = [xp1+vopt1[0], yp1+vopt1[1]]
            
        # clip or extend vopt1End if outside or inside of domain
        vopt1End = boundaryProject(triangulation.bdyPoints, coords, [xp1, yp1], vopt1End, cast=True)
        
        # if midpoint is outside, whole vector is in cavity
        if searchTriangle(0.5*(xp1+vopt1End[0]), 0.5*(yp1+vopt1End[1]), triangulation) == -1:
            proposedPoint1 = -1
            proposedPoint1Metric = -1
        else:
            # already segment length < 1
            if lengthInMetric(metric, [xp1, yp1], vopt1End) < 1:
                proposedPoint1 = -1
                proposedPoint1Metric = -1
            else:
                # point in vopt1 direction from p1 having unit metric length
                proposedPoint1, _ = metricUnitPoint(metric, 1, [[xp1, yp1], vopt1End])

                # finding metric at proposed point
                proposedPoint1Metric = metric(proposedPoint1[0], proposedPoint1[1])

    # point 2
    # finding point q1/2 with eigenvector alined to edgenormal at p1/2
    vp2 = _mostAlignedEigenVector(eigenObjectP2['vec1'], eigenObjectP2['vec2'], edgeNormal)
    tp2 = (eigenObjectP2['lambda1'] if vp2 == eigenObjectP2['vec1'] else eigenObjectP2['lambda2'])**(-0.5)
    xq2, yq2 = [xp2 + vp2[0]*tp2, yp2 + vp2[1]*tp2]

    # if q2 outside boundary leave point proposition
    if searchTriangle(xq2, yq2, triangulation) == -1:
        proposedPoint2 = -1
        proposedPoint2Metric = -1
    else:
        # finding metric at q2
        metricQ2 = metric(xq2, yq2)

        # finding eigvec at q2
        eigenObjectQ2 = eigen(metricQ2)

        # finding aligned eigenvec at q2 with edgenormal
        vq2 = _mostAlignedEigenVector(eigenObjectQ2['vec1'], eigenObjectQ2['vec2'], edgeNormal)

        # average both eigenvectors vp2 and vq2
        vpq2 = [0.5*(vp2[0] + vq2[0]), 0.5*(vp2[1] + vq2[1])]

        # metric anisotropic ratio r2
        r2 = (max(eigenObjectP2['lambda1'], eigenObjectP2['lambda2'])/min(eigenObjectP2['lambda1'], eigenObjectP2['lambda2']))**0.5

        beta2 = 1 - max(0, min((r2-rmin)/(rmax-rmin), 1))
        edgeNormalUnit = unitVec(edgeNormal)
        vopt2 = [beta2*edgeNormalUnit[0] + (1-beta2)*vpq2[0], beta2*edgeNormalUnit[1] + (1-beta2)*vpq2[1]]
        # vopt2 = [beta2*edgeNormalUnit[0] + (1-beta2)*vp2[0], beta2*edgeNormalUnit[1] + (1-beta2)*vp2[1]] # vopt = beta*normal + (1-beta)*vp

        vopt2End = [xp2+vopt2[0], yp2+vopt2[1]]
        
        # clip or extend vopt2End if outside or inside of domain
        vopt2End = boundaryProject(triangulation.bdyPoints, coords, [xp2, yp2], vopt2End, cast=True)
        
        # if midpoint is outside, whole vector is in cavity
        if searchTriangle(0.5*(xp2+vopt2End[0]), 0.5*(yp2+vopt2End[1]), triangulation) == -1:
            proposedPoint2 = -1
            proposedPoint2Metric = -1
        else:
            # already segment length < 1
            if lengthInMetric(metric, [xp2, yp2], vopt2End) < 1:
                proposedPoint2 = -1
                proposedPoint2Metric = -1
            else:
                # point in vopt2 direction from p2 having unit metric length
                proposedPoint2, _ = metricUnitPoint(metric, 1, [[xp2, yp2], vopt2End])

                # finding metric at proposed point
                proposedPoint2Metric = metric(proposedPoint2[0], proposedPoint2[1])

    return [proposedPoint1, proposedPoint2, proposedPoint1Metric, proposedPoint2Metric]

# wrapper for parallelizing findCandidatePoints
def _findCandidatePointsPar(candidateTri: int, candidateEdge: int, triangulation: Triangulation) -> tuple:
     
    segStartNodeIdx = triangulation.triangles[candidateTri+candidateEdge]
    segEndNodeIdx = triangulation.triangles[candidateTri+(candidateEdge+1)%3]

    p1, p2, metricP1, metricP2 = findCandidatePoints(triangulation, segStartNodeIdx, segEndNodeIdx, triangulation.metric)

    return p1, p2, metricP1, metricP2, candidateTri, candidateEdge

# Helper function for each task of findCandidatePoints, which will be passed to the pool
def _processTask(candidateTri: int, candidateEdges: List[int], triangulation: Triangulation):
    return [_findCandidatePointsPar(candidateTri, edge, triangulation) for edge in candidateEdges]

def averagePoints(
        triangulation: Triangulation,
        proposedPoints: List[List[float]], 
        proposedPointsMetric: List[List[float]],
        proposedPointsOrigin: List[int],
        originTri: List[int]) -> List[List[float]]:
    """
    Averages out candidate points originating from the same point. This function is used when the advancing-point type 
    point placement is employed.

    Parameters:
        triangulation (Triangulation): An object representing the triangulated mesh. 
            It must have the following attributes:
            - coordinates: List of vertex coordinates in the mesh.
            - metricMesh: List of metric tensor values at each vertex.
            - metric: A callable that computes the metric tensor at a given point.
        proposedPoints (List[List[float]]): A list of proposed candidate points, each represented as [x, y] coordinates.
        proposedPointsMetric (List[List[float]]): A list of metric tensor values corresponding to each proposed point.
        proposedPointsOrigin (List[int]): A list of indices representing the origin of each proposed point.
        originTri (List[int]): A list of triangle indices representing origins corresponding to each proposed point.

    Returns:
        List[List[float], List[int], List[List[float]]]: A list containing:
            - proposedPointsAvg (List[List[float]]): A list of averaged candidate points, each represented as [x, y] coordinates.
            - originTri (List[int]): A list of triangle indices for the averaged points.
            - proposedPointsMetricAvg (List[List[float]]): A list of metric tensors for the averaged points.

    """
    coords = triangulation.coordinates
    metricMesh = triangulation.metricMesh
    # averaged out arrays
    proposedPointsAvg = []
    proposedPointsOriginAvg = []
    proposedPointsMetricAvg = []

    # while there is point in stack
    while len(proposedPoints) != 0:
        point = proposedPoints.pop()
        originPoint = proposedPointsOrigin.pop()
        pointMetric = proposedPointsMetric.pop()
        triangle = originTri.pop()
        if any(element == originPoint for element in proposedPointsOriginAvg):
            index = proposedPointsOriginAvg.index(originPoint)
        # if index != -1:
            avgX = 0.5*(proposedPointsAvg[index][0]+point[0])
            avgY = 0.5*(proposedPointsAvg[index][1]+point[1])
            avgPoint = [avgX, avgY]
            # if averaged point is outside domain, averaging is not done
            if searchTriangle(avgX, avgY, triangulation) == -1:
                proposedPointsAvg.append(point)
                proposedPointsOriginAvg.append(originPoint)
                originTri.append(triangle)
                proposedPointsMetricAvg.append(pointMetric)
            else:
                avgMetric = triangulation.metric(avgX, avgY)
                originMetric = [metricMesh[int(originPoint*1.5)], metricMesh[int(originPoint*1.5)+1], metricMesh[int(originPoint*1.5)+2]]
                # if averaged point is close to origin point, averaging is not done
                if lengthInMetricMetric(originMetric, avgMetric, [coords[originPoint], coords[originPoint+1]], avgPoint) > 0.9:
                    proposedPointsAvg[index] = [avgX, avgY]
                    proposedPointsMetricAvg[index] = avgMetric
                else:
                    proposedPointsAvg.append(point)
                    proposedPointsOriginAvg.append(originPoint)
                    originTri.append(triangle)
                    proposedPointsMetricAvg.append(pointMetric)
        else:
            proposedPointsAvg.append(point)
            proposedPointsOriginAvg.append(originPoint)
            originTri.append(triangle)
            proposedPointsMetricAvg.append(pointMetric)

    return [proposedPointsAvg, originTri, proposedPointsMetricAvg]

def rejectClosedPoints(
        triangulation: Triangulation,
        points: List[List[float]],
        pointsMetric: List[List[float]],
        originTri: List[int]) -> List[List[float]]:
    """
    Rejects proposed points that are too close to existing points in the triangulation.

    Parameters:
        triangulation (Triangulation): An object representing the triangulated mesh. 
            It must have the following attributes:
            - coordinates: List of vertex coordinates in the mesh.
            - triangles: List of triangle definitions by vertex indices.
            - adjacents: List of adjacency relationships between triangles.
            - metricMesh: List of metric tensor values at each vertex.
            - metric: A callable that computes the metric tensor at a given point.

        points (List[tuple[float, float]]): A list of proposed points to check against the triangulation.
        pointsMetric (List[List[float]]): A list of metric values for each proposed point.
        originTri (List[int]): A list of triangle indices, indicating the origin of each proposed point.

    Returns:
        List[List[List[float]], List[int], List[List[float]]]: A list containing:
            - proposedPoints (List[List[float]]): A list of candidate points that are not rejected, 
                    each represented as [x, y] coordinates.
            - originTri (List[int]): A list of triangle indices for the accepted points.
            - proposedPointsMetric (List[List[float]]): A list of metric tensors for the accepted points.
    
    """
    coords = triangulation.coordinates
    triangles = triangulation.triangles
    adjacents = triangulation.adjacents
    metricMesh = triangulation.metricMesh
    proposedPoints = []
    proposedPointsMetric = []
    triangle = []

    # for all points
    lenPoints = len(points)
    for i in range(lenPoints):
        isClose = False
        pointToCheck = points[i]
        pointToCheckMetric = pointsMetric[i]
        pointToCheckOrigin = originTri[i]

        # finding triangle containing point
        triangleIndex = searchTriangle(pointToCheck[0], pointToCheck[1], triangulation, pointToCheckOrigin)

        if type(triangleIndex) == list:
            # triangle0
            # for all 3 points of containing triangle
            for j in range(triangleIndex[0], triangleIndex[0]+3):
                # find circling triangles
                circlingTrianglesIdx = ballOfNode(triangulation, triangles[j])

                # for all circling triangles
                lenCirclingTrianglesIdx = len(circlingTrianglesIdx)
                for k in range(lenCirclingTrianglesIdx):
                    # take all 3 points of that circling triangle
                    for l in range(3):
                        point = [coords[triangles[circlingTrianglesIdx[k]+l]], coords[triangles[circlingTrianglesIdx[k]+l]+1]]
                        pointMetric = [metricMesh[int(triangles[circlingTrianglesIdx[k]+l]*1.5)], metricMesh[int(triangles[circlingTrianglesIdx[k]+l]*1.5)+1], metricMesh[int(triangles[circlingTrianglesIdx[k]+l]*1.5)+2]]
                        # if point is close to current circling triangle point, break
                        if lengthInMetricMetric(pointMetric, pointToCheckMetric, point, pointToCheck) < 1/2**0.5:
                            isClose = True
                            # this specific structure is required to break out of the outer loop
                            break
                    
                    if isClose: break

                if isClose: break

            # triangle1
            # for all 3 points of containing triangle
            for j in range(triangleIndex[1], triangleIndex[1]+3):
                # find circling triangles
                circlingTrianglesIdx = ballOfNode(triangulation, triangles[j])

                # for all circling triangles
                lenCirclingTrianglesIdx = len(circlingTrianglesIdx)
                for k in range(lenCirclingTrianglesIdx):
                    # take all 3 points of that circling triangle
                    for l in range(3):
                        point = [coords[triangles[circlingTrianglesIdx[k]+l]], coords[triangles[circlingTrianglesIdx[k]+l]+1]]
                        pointMetric = [metricMesh[int(triangles[circlingTrianglesIdx[k]+l]*1.5)], metricMesh[int(triangles[circlingTrianglesIdx[k]+l]*1.5)+1], metricMesh[int(triangles[circlingTrianglesIdx[k]+l]*1.5)+2]]
                        # if point is close to current circling triangle point, break
                        if lengthInMetricMetric(pointMetric, pointToCheckMetric, point, pointToCheck) < 1/2**0.5:
                            isClose = True
                            # this specific structure is required to break out of the outer loop
                            break

                    if isClose: break

                if isClose: break

        else:
            # for all 3 points of containing triangle
            for j in range(triangleIndex, triangleIndex+3):
                # find circling triangles
                circlingTrianglesIdx = ballOfNode(triangulation, triangles[j])

                # for all circling triangles
                lenCirclingTrianglesIdx = len(circlingTrianglesIdx)
                for k in range(lenCirclingTrianglesIdx):
                    # take all 3 points of that circling triangle
                    for l in range(3):
                        point = [coords[triangles[circlingTrianglesIdx[k]+l]], coords[triangles[circlingTrianglesIdx[k]+l]+1]]
                        pointMetric = [metricMesh[int(triangles[circlingTrianglesIdx[k]+l]*1.5)], metricMesh[int(triangles[circlingTrianglesIdx[k]+l]*1.5)+1], metricMesh[int(triangles[circlingTrianglesIdx[k]+l]*1.5)+2]]
                        # if point is close to current circling triangle point, break
                        if lengthInMetricMetric(pointMetric, pointToCheckMetric, point, pointToCheck) < 1/2**0.5:
                            isClose = True
                            # this specific structure is required to break out of the outer loop
                            break
                    
                    if isClose: break
                
                if isClose: break
        
        if not isClose: 
            proposedPoints.append(pointToCheck)
            proposedPointsMetric.append(pointToCheckMetric)
            triangle.append(pointToCheckOrigin)

    return [proposedPoints, triangle, proposedPointsMetric]

def rejectMutualClosedPoints(
        points: List[List[float]],
        pointsMetric: List[List[float]],
        originTri: List[int]) -> List[List[List[float]]]:
    """
    Rejects proposed candidate points that are too close to already accepted points.

    This function iterates through a list of candidate points and checks if each one is
    too close to any of the already accepted points (based on a specific metric). If a 
    candidate point is found to be too close, it is rejected. Among two candidate points 
    that are close to each other, the one with the larger "sizing" criterion (based on 
    eigenvalues) is kept.

    Parameters:
        points (List[List[float]]): A list of candidate points to check, each represented 
            as [x, y] coordinates.
        pointsMetric (List[List[float]]): A list of metric values for each proposed point, 
            typically representing tensor values.
        originTri (List[int]): A list of indices indicating the triangles or origin of each proposed point.

    Returns:
        List[List[List[float]], List[int], List[List[float]]]: A list containing:
            - proposedPoints (List[List[float]]): A list of accepted points, each represented 
                as [x, y] coordinates.
            - proposedPointsOrigin (List[int]): A list of triangle indices corresponding to 
                the accepted points.
            - proposedPointsMetric (List[List[float]]): A list of metric tensor values for the 
                accepted points.
    """
    proposedPoints = [points[0]]
    proposedPointsOrigin = [originTri[0]]
    proposedPointsMetric = [pointsMetric[0]]

    for i in range(1, len(points)):
        startLeg = i-1
        isClose = False
        pointToCheck1 = points[i]
        pointToCheck2Idx = None
        # for j in range(startLeg, -1, -1):
        #     # if lengthInMetric(metric, points[i], points[j]) < 1/2**0.5:
        #     if lengthInMetricMetric(pointsMetric[i], pointsMetric[j], points[i], points[j]) < 1/2**0.5:
        #         isClose = True
        #         pointToCheck2Idx = j
        #         break

        # for all elements in proposedPoints
        for j in range(len(proposedPoints)):
            # if close to already proposed candidate point
            if lengthInMetricMetric(pointsMetric[i], proposedPointsMetric[j], points[i], proposedPoints[j]) < 1/2**0.5:
                isClose = True
                pointToCheck2Idx = j
                break

        if not isClose: 
            proposedPoints.append(points[i])
            proposedPointsOrigin.append(originTri[i])
            proposedPointsMetric.append(pointsMetric[i])
        else:
            # getting eigenvalues of two points in question
            eigen1 = eigen(pointsMetric[i])
            eigen2 = eigen(proposedPointsMetric[pointToCheck2Idx])
            # reject point having larger sizing criterion
            # if abs(eigen1['lambda1']*eigen1['lambda2']) > abs(eigen2['lambda1']*eigen2['lambda2']):
            #     proposedPoints[pointToCheck2Idx] = points[i]
            #     proposedPointsMetric[pointToCheck2Idx] = pointsMetric[i]
            #     originTri[pointToCheck2Idx] = originTri[i]
            if abs(eigen1['lambda1']*eigen1['lambda2']) > abs(eigen2['lambda1']*eigen2['lambda2']):
                proposedPoints[pointToCheck2Idx] = points[i].copy()
                proposedPointsMetric[pointToCheck2Idx] = pointsMetric[i].copy()
                proposedPointsOrigin[pointToCheck2Idx] = originTri[i]

    return [proposedPoints, proposedPointsOrigin, proposedPointsMetric]

def isSwappingRequired(triangulation: Triangulation, uNode: int, vNode: int, aOpNode: int, bOpNode: int, criterion: str) -> bool:
    """
    Determines if a swap is required between two points in a triangulation based on the specified criterion.

    Parameters:
        triangulation (Triangulation): An object representing the triangulated mesh. 
            It must have the following attributes:
            - coordinates: A list of vertex coordinates in the mesh.
            - metricMesh: A list of metric tensor values at each vertex.
            - triangles: A list of triangles in the mesh.
            - metric: A callable to compute the metric tensor at a given point.
        uNode (int): The index of the first node (uNode) in the current configuration.
        vNode (int): The index of the second node (vNode) in the current configuration.
        aOpNode (int): The index of the auxiliary node aOpNode in the current configuration.
        bOpNode (int): The index of the auxiliary node bOpNode in the current configuration.
        criterion (str): The criterion to determine if a swap is required. It can be one of the following:
            - "min-max-metric": Minimizes the maximum metric angle.
            - "min-max": Minimizes the maximum physical angle.

    Returns:
        bool: `True` if a swap is required, `False` otherwise.
            A swap is required if the maximum angle in the swapped configuration is smaller 
            than the maximum angle in the current configuration.

    Raises:
        RuntimeError: If the provided criterion is unknown.
    """
    # swapping criterion: min-max criterion as defined in barth 1991 paper
    if criterion == "min-max-metric":   # measures metric angle
        # minimizing maximum angle
        # finding all angles in current configuration
        angleAopUV = angleInMetric(triangulation, aOpNode, uNode, vNode)
        angleUVAop = angleInMetric(triangulation, uNode, vNode, aOpNode)
        angleVAopU = angleInMetric(triangulation, vNode, aOpNode, uNode)
        angleBopVU = angleInMetric(triangulation, bOpNode, vNode, uNode)
        angleVUBop = angleInMetric(triangulation, vNode, uNode, bOpNode)
        angleUBopV = angleInMetric(triangulation, uNode, bOpNode, vNode)
        maxAngleCurrent = max(angleAopUV, angleBopVU, angleVAopU, angleUVAop, angleVUBop, angleUBopV)

        # finding all angles in swapped configuration
        angleVAopBop = angleInMetric(triangulation, vNode, aOpNode, bOpNode)
        angleAopBopV = angleInMetric(triangulation, aOpNode, bOpNode, vNode)
        angleBopVopA = angleInMetric(triangulation, bOpNode, vNode, aOpNode)
        angleBopAopU = angleInMetric(triangulation, bOpNode, aOpNode, uNode)
        angleUBopAop = angleInMetric(triangulation, uNode, bOpNode, aOpNode)
        angleAopUBop = angleInMetric(triangulation, aOpNode, uNode, bOpNode)
        maxAngleSwapped = max(angleVAopBop, angleAopBopV, angleBopVopA, angleBopAopU, angleUBopAop, angleAopUBop)

        return False if maxAngleCurrent <= maxAngleSwapped else True
    
    elif criterion == "min-max":    # measures physical angle
        # minimizing maximum angle
        # finding all angles in current configuration
        angleAopUV = angle(triangulation, aOpNode, uNode, vNode)
        angleUVAop = angle(triangulation, uNode, vNode, aOpNode)
        angleVAopU = angle(triangulation, vNode, aOpNode, uNode)
        angleBopVU = angle(triangulation, bOpNode, vNode, uNode)
        angleVUBop = angle(triangulation, vNode, uNode, bOpNode)
        angleUBopV = angle(triangulation, uNode, bOpNode, vNode)
        maxAngleCurrent = max(angleAopUV, angleBopVU, angleVAopU, angleUVAop, angleVUBop, angleUBopV)

        # finding all angles in swapped configuration
        angleVAopBop = angle(triangulation, vNode, aOpNode, bOpNode)
        angleAopBopV = angle(triangulation, aOpNode, bOpNode, vNode)
        angleBopVopA = angle(triangulation, bOpNode, vNode, aOpNode)
        angleBopAopU = angle(triangulation, bOpNode, aOpNode, uNode)
        angleUBopAop = angle(triangulation, uNode, bOpNode, aOpNode)
        angleAopUBop = angle(triangulation, aOpNode, uNode, bOpNode)
        maxAngleSwapped = max(angleVAopBop, angleAopBopV, angleBopVopA, angleBopAopU, angleUBopAop, angleAopUBop)

        return False if maxAngleCurrent <= maxAngleSwapped else True
    
    elif criterion == "delaunay":    # based on delaunay incircle property
        coords = triangulation.coordinates
        # checking delaunay criterion
        if inCircle(coords[aOpNode], coords[aOpNode+1], coords[uNode], coords[uNode+1], \
                    coords[vNode], coords[vNode+1], coords[bOpNode], coords[bOpNode+1]):
            return True

        return False
    
    else:
        raise RuntimeError(f"Swapping criterion {criterion} unknown.")

def triangleSubdivide(
        triangulation: Triangulation,
        point: List[float],
        triangleIndex: Union[int, List[int]],
        pointMetric: Optional[List[float]]=None):
    """
    Alters a triangle in the triangulation by inserting a new point and subdividing the triangle.

    Parameters:
        triangulation (Triangulation): An object representing the triangulated mesh. It must have the following attributes:
            - coordinates: A list of vertex coordinates in the mesh.
            - metricMesh: A list of metric tensor values at each vertex.
            - triangles: A list of triangles in the mesh.
            - adjacents: A list of adjacency information for each triangle.
            - flags: A list of flags indicating the status of each triangle.
            - quadTree: A structure to manage the spatial data and insertions of points.
            - adjPointTris: A list of triangles adjacent to each point.
        point (List[float]): The coordinates of the new point to insert, represented as [x, y].
        triangleIndex (Union[int, List[int]]): The index of the triangle to subdivide. If the point is shared
            between two other triangles, this will be a list containing two indices of the triangles.
        pointMetric (Optional[List[float]], optional): The metric tensor values for the new point, represented as
            [m11, m12, m22]. If not provided, not added to `metricMesh`.

    Returns:
        None: The function modifies the triangulation in place by adding new points, triangles, and updating adjacencies.
    """
    # push new point into coordinates array
    triangulation.coordinates.append(point[0])
    triangulation.coordinates.append(point[1])
    newNode = len(triangulation.coordinates)-2

    if pointMetric:
        # push new metric in metricMesh array
        triangulation.metricMesh.append(pointMetric[0])
        triangulation.metricMesh.append(pointMetric[1])
        triangulation.metricMesh.append(pointMetric[2])
        # compute metricLog and push 
        metricLog = logMetric([pointMetric[0], pointMetric[1], pointMetric[2]])
        triangulation.metricLog.append(metricLog[0])
        triangulation.metricLog.append(metricLog[1])
        triangulation.metricLog.append(metricLog[2])

    # if bl elements are there, extend their flags list
    if triangulation.blFlags:
        triangulation.blFlags.append(1)
        triangulation.blFlags.append(1)

    # if given point is on the shared edge
    if type(triangleIndex) == list:
        # _______________________________________________________________
        # Subdivision on edge:
        # 
        #      vIdx.vAdjIdx                    vIdx.vAdjIdx            
        #         /|\                             /|\                     
        #        / | \                           / | \      
        #       /  |  \                         /  |  \                 
        #  aTri/   |   \bTri               aTri/   |   \bTri          
        #     /    |    \     subdivision     /____|p___\            
        # aIdx\    +p   /bIdx     =>      aIdx\    |    /bIdx        
        #      \   |   /                       \   |   /                
        #       \  |  /                         \  |  /            
        #        \ | /                           \ | /                
        #         \|/                             \|/                   
        #      uIdx.uAdjIdx                    uIdx.uAdjIdx                          
        # Triangles:
        #   [uva ... ubv]         =>     [pva ... pbv ... aupbpu]
        # _______________________________________________________________

        # storing triangle structure before subdivision
        aNodes = [triangulation.triangles[triangleIndex[0]], triangulation.triangles[triangleIndex[0]+1], triangulation.triangles[triangleIndex[0]+2]]
        bNodes = [triangulation.triangles[triangleIndex[1]], triangulation.triangles[triangleIndex[1]+1], triangulation.triangles[triangleIndex[1]+2]]
        # finding intersection of both triangles
        diagonalNodes = list(set(aNodes) & set(bNodes))
        # finding opposite nodes
        # diff = list(set(aNodes) - set(diagonalNodes))
        # if len(diff) == 0:
        #     print("Degenerate triangle detected:", aNodes, diagonalNodes)
        #     return
        aOpNode = list(set(aNodes) - set(diagonalNodes))[0]
        bOpNode = list(set(bNodes) - set(diagonalNodes))[0]
        aOpNodeIndex = aNodes.index(aOpNode)
        bOpNodeIndex = bNodes.index(bOpNode)
        uNode = triangulation.triangles[triangleIndex[0]+(aOpNodeIndex+1)%3]
        vNode = triangulation.triangles[triangleIndex[0]+(aOpNodeIndex+2)%3]
        auAdj = triangulation.adjacents[triangleIndex[0]+aOpNodeIndex]
        ubAdj = triangulation.adjacents[triangleIndex[1]+(bOpNodeIndex+2)%3]

        # shifting u to newNode
        triangulation.triangles[triangleIndex[0]+(aOpNodeIndex+1)%3] = newNode
        triangulation.triangles[triangleIndex[1]+(bOpNodeIndex+2)%3] = newNode

        # adding new triangle along au
        triangulation.triangles.append(aOpNode)
        triangulation.triangles.append(uNode)
        triangulation.triangles.append(newNode)

        # adding new triangle along ub
        triangulation.triangles.append(bOpNode)
        triangulation.triangles.append(newNode)
        triangulation.triangles.append(uNode)

        # changing adjacents of old two triangles
        triangulation.adjacents[triangleIndex[0]+aOpNodeIndex] = len(triangulation.triangles)-6
        triangulation.adjacents[triangleIndex[1]+(bOpNodeIndex+2)%3] = len(triangulation.triangles)-3

        # adding adjacents of au triangle
        triangulation.adjacents.append(auAdj)
        triangulation.adjacents.append(len(triangulation.triangles)-3)
        triangulation.adjacents.append(triangleIndex[0])

        # adding adjacents of ub triangle
        triangulation.adjacents.append(triangleIndex[1])
        triangulation.adjacents.append(len(triangulation.triangles)-6)
        triangulation.adjacents.append(ubAdj)

        # for auAdj triangle, setting new au triangle as adjacent
        if auAdj != -1:
            for i in range(auAdj, auAdj+3):
                if triangulation.adjacents[i] == triangleIndex[0]:
                    triangulation.adjacents[i] = len(triangulation.triangles)-6

        # for ubAdj triangle, setting new ub triangle as adjacent
        if ubAdj != -1:
            for i in range(ubAdj, ubAdj+3):
                if triangulation.adjacents[i] == triangleIndex[1]:
                    triangulation.adjacents[i] = len(triangulation.triangles)-3

        if triangulation.flags:
            # turning on flags of old triangles
            triangulation.flags[int(triangleIndex[0]/3)] = 1
            triangulation.flags[int(triangleIndex[1]/3)] = 1

            # adding flags for newly created triangles
            triangulation.flags.append(1)
            triangulation.flags.append(1)

        # insert new point to quadtree and attach new triangle here
        triangulation.quadTree.insert(len(triangulation.coordinates)-2)
        triangulation.adjPointTris.append(len(triangulation.triangles)-3)

        # change old triangle attached to u point
        if triangulation.adjPointTris[uNode//2] == triangleIndex[0] \
            or triangulation.adjPointTris[uNode//2] == triangleIndex[1]:
            triangulation.adjPointTris[uNode//2] = len(triangulation.triangles)-6
    
    else:
        # _______________________________________________________________
        # Subdivision:
        # 
        #        o                              o                       
        #       / \                            /|\         
        #      /   \        subdivision       / | \        
        #     /  .  \           =>           /  .p \     
        #    /  p    \                      /  / \  \    
        #   /         \                    / /     \ \  
        #  .___________.                  ./_________\. 
        # u             v                u             v
        #
        # Triangles:
        #   [...uvo...]         =>       [...uvp...vop...oup]
        # _______________________________________________________________ 

        # storing triangle structure before subdivision
        uNode = triangulation.triangles[triangleIndex]
        vNode = triangulation.triangles[triangleIndex+1]
        oppositeNode = triangulation.triangles[triangleIndex+2]
        voAdj = triangulation.adjacents[triangleIndex+1]
        ouAdj = triangulation.adjacents[triangleIndex+2]

        # changing oppositeNode to new inserted point
        triangulation.triangles[triangleIndex+2] = newNode

        # adding new triangle along vo
        triangulation.triangles.append(vNode)
        triangulation.triangles.append(oppositeNode)
        triangulation.triangles.append(newNode)
        
        # adding new triangle along ou
        triangulation.triangles.append(oppositeNode)
        triangulation.triangles.append(uNode)
        triangulation.triangles.append(newNode)

        # adding v-new adjacent and u-new adjacent
        triangulation.adjacents[triangleIndex+1] = len(triangulation.triangles)-6
        triangulation.adjacents[triangleIndex+2] = len(triangulation.triangles)-3

        # adding adjacents for new triangle along vo
        triangulation.adjacents.append(voAdj)
        triangulation.adjacents.append(len(triangulation.triangles)-3)
        triangulation.adjacents.append(triangleIndex)

        # adding adjacents for new triangle along ou
        triangulation.adjacents.append(ouAdj)
        triangulation.adjacents.append(triangleIndex)
        triangulation.adjacents.append(len(triangulation.triangles)-6)

        # for voAdj triangle, setting new vo triangle as adjacent
        if voAdj != -1:
            for i in range(voAdj, voAdj+3):
                if triangulation.adjacents[i] == triangleIndex:
                    triangulation.adjacents[i] = len(triangulation.triangles)-6

        # for ouAdj triangle, setting new ou triangle as adjacent
        if ouAdj != -1:
            for i in range(ouAdj, ouAdj+3):
                if triangulation.adjacents[i] == triangleIndex:
                    triangulation.adjacents[i] = len(triangulation.triangles)-3

        if triangulation.flags:
            # turning on flag of old triangle
            triangulation.flags[int(triangleIndex/3)] = 1

            # adding flags for newly created triangles
            triangulation.flags.append(1)
            triangulation.flags.append(1)

        # insert new point to quadtree and attach new triangle here
        triangulation.quadTree.insert(len(triangulation.coordinates)-2)
        triangulation.adjPointTris.append(triangleIndex)

        # change old triangle attached to opposite point
        if triangulation.adjPointTris[oppositeNode//2] == triangleIndex:
            triangulation.adjPointTris[oppositeNode//2] = len(triangulation.triangles)-6

    # incrementally update element ADT: the subdivided triangle(s) changed
    # bounding box, and 2 new triangles were appended to the end
    nTriangles = len(triangulation.triangles)
    if type(triangleIndex) == list:
        triangulation.elementADT.update(triangleIndex[0], elementBoundingBox(triangulation, triangleIndex[0]))
        triangulation.elementADT.update(triangleIndex[1], elementBoundingBox(triangulation, triangleIndex[1]))
    else:
        triangulation.elementADT.update(triangleIndex, elementBoundingBox(triangulation, triangleIndex))
    triangulation.elementADT.insert(elementBoundingBox(triangulation, nTriangles-6), nTriangles-6)
    triangulation.elementADT.insert(elementBoundingBox(triangulation, nTriangles-3), nTriangles-3)


# swaps diagonal of given quadrilateral defined by segment between node Indexes uIdx and vIdx
def swapDiagonal(triangulation: Triangulation, uIdx: int, bIdx: int) -> None:
    """
    Swaps the diagonal of the quadrilateral formed by two adjacent triangles in the triangulation.

    Parameters:
        triangulation (Triangulation): An object representing the triangulated mesh. 
            It must have the following attributes:
            - coordinates: A flat list of vertex coordinates.
            - triangles: Triangle vertex indices grouped in sets of 3.
            - adjacents: Adjacency indices per triangle edge.
            - adjPointTris: Triangle index adjacent to each vertex.
            - flags: Flags associated with each triangle.
        uIdx (int): Index in the triangle list representing one end of the shared diagonal to be swapped.
        bIdx (int): Index in the triangle list representing the corresponding position in the adjacent triangle.

    Returns:
        None: The triangulation object is modified in-place.
    """
    # _______________________________________________________________
    # Swapping mechanism:
    # 
    #     vIdx/\vAdjIdx                  vIdx/\vAdjIdx  
    #        /||\                           /  \        
    #       / || \                         /    \       
    #  aTri/  ||  \bTri                   / aTri \      
    #     /   ||   \        flip         /________\     
    # aIdx\   ||   /bIdx     =>      aIdx\————————/bIdx 
    #      \  ||  /                       \ bTri /      
    #       \ || /                         \    /       
    #        \||/                           \  /        
    #     uIdx\/uAdjIdx                  uIdx\/uAdjIdx
    # 
    # Triangles:
    #   [uva ... ubv]        =>         [bva ... uba]
    # _______________________________________________________________

    aTriStartIdx = uIdx - uIdx%3
    bTriStartIdx = bIdx - bIdx%3
    aIdx = aTriStartIdx + ((uIdx%3) + 2)%3
    uNode = triangulation.triangles[uIdx]
    vNode = triangulation.triangles[aTriStartIdx + (uIdx%3+1)%3]

    # swapping triangles
    triangulation.triangles[uIdx] = triangulation.triangles[bIdx]
    triangulation.triangles[bTriStartIdx + ((bIdx%3)+1)%3] = triangulation.triangles[aIdx]

    # swapping adjacents
    bvAdj = triangulation.adjacents[bIdx]
    auAdj = triangulation.adjacents[aIdx]

    triangulation.adjacents[uIdx] = bvAdj
    triangulation.adjacents[bTriStartIdx + ((bIdx%3)+1)%3] = auAdj

    triangulation.adjacents[bIdx] = aTriStartIdx
    triangulation.adjacents[aIdx] = bTriStartIdx
    
    # changing adjacents of surrounding adjacents
    if auAdj != -1:
        for i in range(auAdj, auAdj+3):
            if triangulation.adjacents[i] == aTriStartIdx:
                triangulation.adjacents[i] = bTriStartIdx
    
    if bvAdj != -1:
        for i in range(bvAdj, bvAdj+3):
            if triangulation.adjacents[i] == bTriStartIdx:
                triangulation.adjacents[i] = aTriStartIdx

    # swapping attached triangles at u and v
    if triangulation.adjPointTris[uNode//2] == aTriStartIdx:
        triangulation.adjPointTris[uNode//2] = bTriStartIdx

    if triangulation.adjPointTris[vNode//2] == bTriStartIdx:
        triangulation.adjPointTris[vNode//2] = aTriStartIdx
    
    # activating flags if available
    if triangulation.flags:
        triangulation.flags[aTriStartIdx//3] = 1
        triangulation.flags[bTriStartIdx//3] = 1

    # incrementally update element ADT: both triangles changed bounding box
    triangulation.elementADT.update(aTriStartIdx, elementBoundingBox(triangulation, aTriStartIdx))
    triangulation.elementADT.update(bTriStartIdx, elementBoundingBox(triangulation, bTriStartIdx))

def localReconnection(
        triangulation: Triangulation,
        criterion: str,
        skipTriangleFlags: Optional[list[int]]=None) -> int:
    """
    Performs local reconnection (edge flipping) on internal mesh edges to improve mesh quality
    based on a given swapping criterion.

    Parameters:
        triangulation (Triangulation): An object representing the triangulated mesh. 
            It must have the following attributes:
            - coordinates: A flat list of vertex coordinates.
            - triangles: Triangle vertex indices grouped in sets of 3.
            - adjacents: Adjacency indices per triangle edge.
            - flags: Flags indicating active triangles.
            - adjPointTris: Triangle indices associated with each point.
        criterion (str): The edge-flipping criterion, e.g., "min-max" or "min-max-metric".
        skipTriangleFlags (Optional[List[int]], optional): List of binary flags indicating 
            which triangles should be considered for edge flipping. Defaults to None.
    
    Returns:
        int: The number of successful edge swaps performed during the reconnection process.

    Raises:
        RuntimeError: If the reconnection process terminates without emptying the internal edge stack 
            within the predefined maximum swap limit.
    """
    coords = triangulation.coordinates

    internalEdgesIdx = []
    # collect all the edges of active triangles
    # hash table to store edges with same keys
    nPoints = len(triangulation.coordinates)//2
    hashTable = [0]*nPoints
    lenTriangles = len(triangulation.triangles)
    for i in range(0, lenTriangles, 3):
        # if flags exist, see if triangle active otherwise ignore flags
        if not triangulation.flags or triangulation.flags[int(i/3)] == 1:
        # for all three edges
            for j in range(3):
                # skip edges on boundary hull
                if triangulation.adjacents[i+j] == -1: continue

                u = triangulation.triangles[i+j]
                v = triangulation.triangles[i+(j+1)%3]
                uIdx = i+j
                # key to index in hash table
                key = (u+v)%nPoints

                # if new edge at this key, store in hash table
                if hashTable[key] == 0:
                    hashTable[key] = [(i, j, min(u, v))]
                    # also add global edge index
                    internalEdgesIdx.append(uIdx)
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
                        # also add global edge index
                        internalEdgesIdx.append(uIdx)

    MAX_SWAPS_LIMIT = 50000

    nSwaps = 0
    while nSwaps < MAX_SWAPS_LIMIT:

        # if stack empty, break
        if len(internalEdgesIdx) == 0:
            return nSwaps

        # edge start index
        uIdx = internalEdgesIdx.pop()

        # a triangle start index
        aTriStartIdx = uIdx-(uIdx%3)
        vIdx = aTriStartIdx + (uIdx+1)%3

        # constructing quadrilateral from u and v nodes
        uNode = triangulation.triangles[uIdx]
        vNode = triangulation.triangles[vIdx]

        # finding a node
        aOpNodeIdx = aTriStartIdx + ((uIdx%3) + 2)%3
        aOpNode = triangulation.triangles[aOpNodeIdx]

        # finding b node
        bTriStartIdx = triangulation.adjacents[uIdx]
        # adjacent is not there, leave this edge
        if bTriStartIdx == -1: continue

        # skip edge if atleast one triangle is off in skip list
        if skipTriangleFlags:
            if skipTriangleFlags[aTriStartIdx//3] == 0 or skipTriangleFlags[bTriStartIdx//3] == 0:
                continue

        # finding bOpNode in adjacent triangle
        if triangulation.triangles[bTriStartIdx] == uNode:
            bOpNode = triangulation.triangles[bTriStartIdx+1]
            bOpNodeIdx = bTriStartIdx+1
        elif triangulation.triangles[bTriStartIdx+1] == uNode:
            bOpNode = triangulation.triangles[bTriStartIdx+2]
            bOpNodeIdx = bTriStartIdx+2
        elif triangulation.triangles[bTriStartIdx+2] == uNode:
            bOpNode = triangulation.triangles[bTriStartIdx]
            bOpNodeIdx = bTriStartIdx
        else:
            raise RuntimeError

        # check if quadrilateral is convex or not
        if not doSegmentsIntersect(coords[uNode], coords[uNode+1], coords[vNode], coords[vNode+1],
            coords[aOpNode], coords[aOpNode+1], coords[bOpNode], coords[bOpNode+1]):
            continue

        # swap diagonal if required and push other adjacent edges to stack
        if isSwappingRequired(triangulation, uNode, vNode, aOpNode, bOpNode, criterion):
            swapDiagonal(triangulation, uIdx, bOpNodeIdx)

            uIdxAdj = bTriStartIdx + (bOpNodeIdx-1)%3
            vIdxAdj = bTriStartIdx + (bOpNodeIdx+1)%3

            # add quadrilateral edges to internalEdgesIdx
            # Note that the insertion edge starting indexes are mapped once the middle edge is swapped
            internalEdgesIdx.insert(0, uIdx)
            internalEdgesIdx.insert(0, vIdx)
            internalEdgesIdx.insert(0, vIdxAdj)
            internalEdgesIdx.insert(0, uIdxAdj)

            nSwaps += 1
        
    # if for loop completes all iterations while having nonzero stack
    raise RuntimeError('Local reconnection terminated at incomplete stage, increase max swap limit.')

# this function generates new mesh based on given mesh by adding points 
# according to aflr algorithm
def aflrAdapt(
        in2dPath: str,
        bgMeshFunction: Callable[[float, float], List[float]],
        startTriangulation: Optional[Triangulation]=None,
        iterations: Optional[int]=None,
        runParallel: Optional[bool]=False) -> Triangulation:
    """
    Performs the AFLR (Advancing Front and Local Reconnection) algorithm.

    The function starts with an initial triangulation, and iteratively refines the mesh by adding new points
    based on a background metric function, ensuring that the resulting mesh satisfies the criteria of the AFLR algorithm.

    Parameters:
        in2dPath (str): Path to the `.in2d` geometry input file for the mesh initialization.
        bgMeshFunction (Callable[[float, float], List[float]]): A function that defines the background mesh metric.
        startTriangulation (Optional[Triangulation], optional): The initial triangulation to start from. If None, a new triangulation is created.
        iterations (Optional[int], optional): The number of AFLR iterations to perform. Defaults to 40 if None.
        runParallel (Optional[bool], optional): If True, the point proposal process will be parallelized. Defaults to False.

    Returns:
        Triangulation: The updated triangulation after AFLR adaptation.

    Raises:
        RuntimeError: If flat triangles are found or if the mesh is not adapted successfully after the maximum iteration limit.
    """
    print("Starting AFLR...")
    if startTriangulation == None:
        # initializing new mesh
        triangulation = Triangulation.fromIn2D(in2dPath, bgMeshFunction)

        # discretize boundary according to metric
        boundaryMesh(triangulation)
        # once boundary mesh done, set triangulation search epsilon back to default
        triangulation.epsilon = 1e-10
        # local reconnection to optimize quality of starting triangulation
        localReconnection(triangulation, "min-max")

        # check for flat triangles
        checkFlatTriangles(triangulation.coordinates, triangulation.triangles)

        print(f"\rBoundary discretized.\x1B[0K")

    else:
        # start from the given triangulation
        triangulation = startTriangulation

        # print("Using provided starting triangulation.")

        # once boundary mesh done, set triangulation search epsilon back to default
        triangulation.epsilon = 1e-10

    # building flags for internal use in aflr based on target metric sizing of triangles
    triangulation.flags = metricSizingFlags(triangulation)

    # alfr loop
    itr = iterations if iterations != None else 40
    for iteration in range(itr):
        startTime = time.time()
        print('Iteration = {}'.format(iteration))
        triangles = triangulation.triangles
        coords = triangulation.coordinates
        adjacents = triangulation.adjacents
        metricMesh = triangulation.metricMesh

        # finding active triangles which can have candidate points
        candidateActiveTriangles = []
        lenFlags = len(triangulation.flags)
        for i in range(lenFlags):
            # if that triangle is active
            if triangulation.flags[i] == 1:
                triangleIndex = i*3
                for j in range(triangleIndex, triangleIndex+3):
                    # if triangle is on convex hull, consider it as a candidate
                    if adjacents[j] == -1:
                        candidateActiveTriangles.append(triangleIndex)
                        break
                    # if triangle has an adjacent which is off, consider it as a candidate
                    if triangulation.flags[int(adjacents[j]/3)] == 0:
                        candidateActiveTriangles.append(triangleIndex)
                        break
        print('\tCollected candidate triangles.')
        # for candidate active triangles find candidate edge
        candidateEdges = []
        for triangleIndex in candidateActiveTriangles:
            # for given triangle index, store candidate edges in array
            edges = []
            # for all three edges
            for i in range(triangleIndex, triangleIndex+3):
                # if adjacent is on boundary, push that edge
                if adjacents[i] == -1:
                    edges.append(i-triangleIndex)
                    # drawsegment
                # if adjacent is not active, push that edge
                if triangulation.flags[int(adjacents[i]/3)] == 0:
                    edges.append(i-triangleIndex)
                    # drawsegment
            candidateEdges.append(edges)
        print('\tCollected candidate edges.')
        # print(time.time() - startTime)

        # for all candidateActiveTriangles propose points from all candidateEdges 
        # of that triangle
        proposedPoints = []
        proposedPointsOrigin = []
        originTri = []
        proposedPointsMetric = []
        # for all candidate triangles
        lenCandidateActiveTriangles = len(candidateActiveTriangles)
        startTime = time.time()
        if not runParallel:
            progress = 0
            progressP = None
            for i in range(lenCandidateActiveTriangles):
                # for all candidate edges of that candidate triangle
                lenCandidateEdges = len(candidateEdges[i])
                for j in range(lenCandidateEdges):
                    # print(time.time() - startTime, 'bef')

                    segStartNodeIdx = triangles[candidateActiveTriangles[i]+candidateEdges[i][j]]
                    segEndNodeIdx = triangles[candidateActiveTriangles[i]+(candidateEdges[i][j]+1)%3]
                    # if candidateActiveTriangles[i] == 165: print(segStartNodeIdx, segEndNodeIdx)
                    # propose new points from given edge
                    p1, p2, metricP1, metricP2 = findCandidatePoints(triangulation, segStartNodeIdx, segEndNodeIdx, triangulation.metric)
                    # print(time.time() - startTime, 'aft')
                    progress = i/lenCandidateActiveTriangles
                    if (int(progress*10) != progressP):
                        print(f"\r\tProposing points; Progress: {int(progress*100)}%", end="")
                        progressP = int(progress*10)

                    # if p1 valid then only consider
                    if p1 != -1:
                        proposedPoints.append(p1)
                        proposedPointsOrigin.append(triangles[candidateActiveTriangles[i]+candidateEdges[i][j]])
                        originTri.append(candidateActiveTriangles[i])
                        proposedPointsMetric.append(metricP1)

                    # if p2 valid then only consider
                    if p2 != -1:
                        proposedPoints.append(p2)
                        proposedPointsOrigin.append(triangles[candidateActiveTriangles[i]+(candidateEdges[i][j]+1)%3])
                        originTri.append(candidateActiveTriangles[i])
                        proposedPointsMetric.append(metricP2)
        else:
            # parallelized version of finding candidate points
            with multiprocessing.Pool(processes=12) as pool:
                # map function working on each element of candidateActiveTriangles and each list of candidateEdges at i
                results = pool.starmap(_processTask, [(candidateActiveTriangles[i], candidateEdges[i], triangulation) for i in range(lenCandidateActiveTriangles)])

                # unpacking results
                # for each triangle
                for result in results:
                    # for each edge in triangle
                    for elem in result:
                        # unpack tuple of results
                        p1, p2, metricP1, metricP2, candTri, candEdge = elem
                        
                        # if p1 valid, push it further
                        if p1 != -1:
                            proposedPoints.append(p1)
                            proposedPointsOrigin.append(triangles[candTri+candEdge])
                            originTri.append(candTri)
                            proposedPointsMetric.append(metricP1)
                        # if p2 valid, push it further
                        if p2 != -1:
                            proposedPoints.append(p2)
                            proposedPointsOrigin.append(triangles[candTri+(candEdge+1)%3])
                            originTri.append(candTri)
                            proposedPointsMetric.append(metricP2)

        print('\r\tPoints proposed.                 ')
        # print(time.time() - startTime)
        # if iteration == 0:
        #     ps = [x for xs in proposedPoints for x in xs]
        #     plt.cla()
        #     ax = drawTriangulation(triangulation, flags=triangulation.flags)
        #     plt.scatter(ps[::2], ps[1::2])
        #     plt.show()
        
        # averaging proposed points originating from same points
        proposedPoints, originTri, proposedPointsMetric = averagePoints(triangulation, proposedPoints, proposedPointsMetric, proposedPointsOrigin, originTri)
        print('\tPoints averaged.')

        # if iteration == 0:
        #     ps = [x for xs in proposedPoints for x in xs]
        #     plt.cla()
        #     ax = drawTriangulation(triangulation, flags=triangulation.flags)
        #     plt.scatter(ps[::2], ps[1::2])
        #     plt.show()

        # reject proposed points which are closed to existing triangulation point
        proposedPoints, originTri, proposedPointsMetric = rejectClosedPoints(triangulation, proposedPoints, proposedPointsMetric, originTri)
        print('\tPoints rejected.')
        # print(len(proposedPoints))
        # if iteration == 0:
        #     ps = [x for xs in proposedPoints for x in xs]
        #     plt.cla()
        #     ax = drawTriangulation(triangulation, flags=triangulation.flags)
        #     plt.scatter(ps[::2], ps[1::2])
        #     plt.show()
        # if there are no further proposed points
        if len(proposedPoints) == 0:
            print('\tNumber of proposed points = ', len(proposedPoints))
            # do final pass of local reconnection
            swaps = localReconnection(triangulation, criterion="min-max-metric")

            # update the flags
            triangulation.flags = metricSizingFlags(triangulation)
            print("\tNo further points proposed for insertion.")
            print('\tTotal {0} points and {1} elements.'.format(len(triangulation.coordinates)//2, len(triangulation.triangles)//3))
            print("AFLR stopped.")
            # break out of the AFLR loop
            break
        
        # reject candidate points which are too close to other candidate points
        proposedPoints, originTri, proposedPointsMetric = rejectMutualClosedPoints(proposedPoints, proposedPointsMetric, originTri)
        print('\tCandidate points mutually rejected.')
        # print(np.array(proposedPoints))
        # if iteration == 0:
        #     ps = [x for xs in proposedPoints for x in xs]
        #     from .triangulationDraw import drawMetricBalls
        #     plt.cla()
        #     ax = drawTriangulation(triangulation)
        #     drawMetricBalls(triangulation, bgMeshFunction, ax)
        #     plt.scatter(ps[::2], ps[1::2])
        #     plt.show()

        # insert proposed points by direct subdivision
        lenProposedPoints = len(proposedPoints)
        for i in range(lenProposedPoints):
            triangleIndex = searchTriangle(proposedPoints[i][0], proposedPoints[i][1], triangulation, originTri[i])
            # if triangle is frozen BL layer element, leave it
            if triangulation.blFlags and triangulation.blFlags[triangleIndex//3] == 0: continue
            # only subdivide if triangle index is valid
            if triangleIndex != -1:
                triangleSubdivide(triangulation, proposedPoints[i], triangleIndex, proposedPointsMetric[i]) 
        print('\tTriangles subdivided.')

        # print(time.time() - startTime)

        # do local reconnection pass
        # print(time.time() - startTime, 'bf')
        swaps = localReconnection(triangulation, criterion="min-max-metric", skipTriangleFlags=triangulation.blFlags)
        print(f'\tLocal reconnection completed with {swaps} swaps.')
        if swaps == 0:
            break
        # print(time.time() - startTime, 'aft')
        # print(swaps, iteration, len(proposedPoints))
        print('\tTotal {0} points and {1} elements'.format(len(triangulation.coordinates)//2, len(triangulation.triangles)//3))
        # update flags
        triangulation.flags = metricSizingFlags(triangulation)

        # # make adjacents
        # triangulation.adjacents = search.adjacencyList(triangulation.triangles)

    # check for flat triangles
    checkFlatTriangles(triangulation.coordinates, triangulation.triangles)
    
    return triangulation
    


# helpers
# checking code to find flat triangles based on determinant
def checkFlatTriangles(coords: List[float], triangles: List[int]) -> List[int]:
    """
    Checks for flat triangles in a given triangulation based on the determinant of the triangle's area.

    A flat triangle is considered when the determinant (area) of the triangle is near zero, 
    which indicates that the triangle's vertices are almost collinear.

    Parameters:
        coords (List[float]): A list of coordinates for the nodes, where the x and y coordinates 
                               for each node are stored consecutively [x, y, x, y, ...].
        triangles (List[int]): A list of triangle indices, where each set of three consecutive 
                                values corresponds to a triangle, and each value represents a node index.

    Returns:
        None.

    Raises:
        RuntimeError: If flat triagles are found

    Notes:
        - The function calculates the determinant of the triangle's area using the formula:
          det = (ax-cx)*(by-cy) - (ay-cy)*(bx-cx)
        - The result is normalized by the maximum edge length squared to make the check scale-independent.
        - Triangles with an absolute determinant value less than or equal to `EPSILON` are considered flat.

    """
    EPSILON = 1e-15
    indexes = []
    lenTriangles = len(triangles)
    for i in range(0, lenTriangles, 3):
        ax = coords[triangles[i]]
        ay = coords[triangles[i]+1]
        bx = coords[triangles[i+1]]
        by = coords[triangles[i+1]+1]
        cx = coords[triangles[i+2]]
        cy = coords[triangles[i+2]+1]
        det = (ax-cx)*(by-cy) - (ay-cy)*(bx-cx)

        # normalize by max edge length squared for scale independence
        l1 = ((ax-bx)**2 + (ay-by)**2)
        l2 = ((bx-cx)**2 + (by-cy)**2)
        l3 = ((cx-ax)**2 + (cy-ay)**2)
        det /= max(l1, l2, l3)

        if (abs(det) <= EPSILON): indexes.append(i)
    
    if len(indexes) > 0:
        # raise RuntimeError("Flat Triangles")
        print("Flat triangles found.")
        

    return

def _mostAlignedEigenVector(vec1: List[float], vec2: List[float], directionVector: List[float]) -> List[float]:
    """
    Finds the vector (either `vec1` or `vec2`) that is most aligned with the `directionVector`.
    
    Parameters:
        vec1 (List[float]): First vector.
        vec2 (List[float]): Second vector.
        directionVector (List[float]): Direction vector to align with.
        
    Returns:
        List[float]: The vector (either `vec1` or `vec2`) that is most aligned with `directionVector`.
    """
    vec1Unit = unitVec(vec1)
    vec2Unit = unitVec(vec2)
    directionVectorUnit = unitVec(directionVector)
    dot1 = dotProduct(vec1Unit, directionVectorUnit)
    dot2 = dotProduct(vec2Unit, directionVectorUnit)

    vec = vec1 if abs(dot1) > abs(dot2) else vec2
    if dotProduct(unitVec(vec), directionVectorUnit) < 0:
        vec[0] = -vec[0]
        vec[1] = -vec[1]
    
    return vec
