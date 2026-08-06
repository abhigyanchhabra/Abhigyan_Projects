import numpy as np
from typing import TYPE_CHECKING, List, Callable, Optional, Dict
if TYPE_CHECKING:
    from .triangulation import Triangulation

from .gaussianquadr import GAUSSIAN_QUADR_XS_64, GAUSSIAN_QUADR_WEIGHTS_64


def boundaryMesh(triangulation: 'Triangulation') -> None:
    """
    Discretizes boundary.

    Parameters:
        triangulation (Triangulation): A triangulation object containing a 'meshObj' attribute with boundary data.
            - meshObj['edgeSegments'] (list[dict]): List of dictionaries defining edge segments, each containing
              'surfid', 'p1', 'p2', etc.

    Returns:
        None
    """
    segs = len(triangulation.meshObj['edgeSegments'])
    currentEdge = 0
    for i in range(segs):
        increment = discretizeSegment(triangulation, currentEdge)
        currentEdge += increment

# def discretizeSegment(triangulation: 'Triangulation', esIdx: int) -> int:
#     """
#     Discretizes a boundary segment into smaller segments if necessary, based on the metric length of the segment.

#     This function takes an edge segment and discretizes it into smaller segments based on the segment length.
#     If the segment is sufficiently short (less than 1 in the metric unit), it remains as a single segment. 
#     Otherwise, it is divided further into smaller segments until the segment length meets the required threshold.

#     Parameters:
#         triangulation (Triangulation): A triangulation object containing:
#             - meshObj['edgeSegments']: List of edge segment dictionaries.
#             - coordinates: List of points for the triangulation.
#             - metric: A callable that computes the metric tensor at a given point.
#             - in2dObj: Contains segment definitions with control points for curves.
#         esIdx (int): The index of the edge segment to discretize.

#     Returns:
#         int: The number of discretized segments.

#     Raises:
#         ValueError: If the segment has zero length (start and end points are the same).
#     """
#     bdyPoints = triangulation.bdyPoints
#     coords = triangulation.coordinates

#     esIdx *= 2

#     segStart = [coords[bdyPoints[esIdx]], coords[bdyPoints[esIdx]+1]]
#     segEnd = [coords[bdyPoints[esIdx+1]], coords[bdyPoints[esIdx+1]+1]]
#     # if 2nd order bezier curve
#     segControl = None
    
#     edgeSeg = triangulation.meshObj['edgeSegments'][esIdx//2]['ednr1']
    
#     if 'p3' in triangulation.in2dObj['segments'][edgeSeg]:
#         pc = triangulation.in2dObj['segments'][edgeSeg]['p2']
#         segControl = [triangulation.in2dObj['points'][pc*2], triangulation.in2dObj['points'][pc*2+1]]
    
#     # calculate total length of given segment
#     L = lengthInMetric(triangulation.metric, segStart, segEnd)

#     # check if segment has same start and end point
#     if L <= 1e-17: raise ValueError("Selected points lead to segment of zero length")


#     # if L<1 already in metricUnit, leave it as it is
#     # else do further discretization of segment
#     if L < 1:
#         # return the number of segments returned
#         return 1
#     else:
#         # remove edgeSegment 
#         segmentObjs = [triangulation.meshObj['edgeSegments'].pop(esIdx//2)]
#         bdyPointsArray = []

#         t = 0
#         tP = None
#         segEndP = segStart.copy()
#         # t for the start on the segment
#         tCurve = segmentObjs[0]['dist1']

#         while(lengthInMetric(triangulation.metric, segEndP, segEnd) > 2**0.5):
#             if segControl:
#                 [x, y], tCurve = metricUnitPoint(triangulation.metric, 1, [segStart, segControl, segEnd], onBoundary=True, tStart=tCurve)
#             else:
#                 [x, y], tCurve = metricUnitPoint(triangulation.metric, 1, [segStart, segEnd], onBoundary=True, tStart=tCurve)
#             t = (((x-segStart[0])**2 + (y-segStart[1])**2)/((segEnd[0]-segStart[0])**2 + (segEnd[1]-segStart[1])**2))**0.5

#             # if last segment metric length less than 0.707, leave it
#             if lengthInMetric(triangulation.metric, [x, y], segEnd) < 1/(2**0.5): break
            
#             if t<1:
#                 if (int(t*10) != tP):
#                     print(f"\rDiscretizing boundary; Progress: {int(t*100)}%\x1B[0K", end='')
#                     tP = int(t*10)

#                 # build new segment object
#                 segObj = {
#                     'surfid': segmentObjs[-1]['surfid'],
#                     'p1': len(coords)//2,
#                     'p2': segmentObjs[-1]['p2'],
#                     'sf1': segmentObjs[-1]['sf1'],
#                     'sf2': segmentObjs[-1]['sf2'],
#                     'ednr1': segmentObjs[-1]['ednr1'],
#                     'dist1': t,
#                     'ednr2': segmentObjs[-1]['ednr2'],
#                     'dist2': segmentObjs[-1]['dist2']
#                 }

#                 # edit previous segment object
#                 segmentObjs[-1]['p2'] = len(coords)//2
#                 segmentObjs[-1]['dist2'] = t

#                 # add new point in bdy segment
#                 bdyPointsArray.append(len(coords))
#                 bdyPointsArray.append(len(coords))

#                 # append new segment object
#                 segmentObjs.append(segObj)

#                 # subdivide containing triangle
#                 coords.append(x)
#                 coords.append(y)

#                 segEndP[0] = x
#                 segEndP[1] = y 

#         # adding back all edge segments
#         triangulation.meshObj['edgeSegments'][esIdx//2:esIdx//2] = segmentObjs
#         bdyPoints[esIdx+1:esIdx+1] = bdyPointsArray
    
#         # return the number of segments returned
#         return len(segmentObjs)

def bdySeg2Triangle(triangulation, p1, p2):
    for triIdx in range(0, len(triangulation.triangles), 3):
        for edge in range(3):
            if triangulation.adjacents[triIdx+edge] == -1:
                if triangulation.triangles[triIdx+edge] == p1 and \
                triangulation.triangles[triIdx+(edge+1)%3] == p2:
                    return triIdx+edge

def discretizeSegment(triangulation: 'Triangulation', esIdx: int) -> int:
    from .edgePrimitive import bdyTriangleSubdivide
    bdyPoints = triangulation.bdyPoints
    coords = triangulation.coordinates

    esIdx *= 2

    segStart = [coords[bdyPoints[esIdx]], coords[bdyPoints[esIdx]+1]]
    segEnd = [coords[bdyPoints[esIdx+1]], coords[bdyPoints[esIdx+1]+1]]
    # if 2nd order bezier curve
    segControl = None
    
    edgeSeg = triangulation.meshObj['edgeSegments'][esIdx//2]['ednr1']
    
    if 'p3' in triangulation.in2dObj['segments'][edgeSeg]:
        pc = triangulation.in2dObj['segments'][edgeSeg]['p2']
        segControl = [triangulation.in2dObj['points'][pc*2], triangulation.in2dObj['points'][pc*2+1]]
    
    # calculate total length of given segment
    L = lengthInMetric(triangulation.metric, segStart, segEnd)

    # check if segment has same start and end point
    if L <= 1e-17: raise ValueError("Selected points lead to segment of zero length")


    # if L<1 already in metricUnit, leave it as it is
    # else do further discretization of segment
    if L < 1:
        # return the number of segments returned
        return 1
    else:
        t = 0
        tP = None
        segEndP = segStart.copy()
        # t for the start on the segment
        tCurve = triangulation.meshObj['edgeSegments'][esIdx//2]['dist1']
        nSegs = 0
        while(lengthInMetric(triangulation.metric, segEndP, segEnd) > 2**0.5):
        # for i in range(2):
            if segControl:
                [x, y], tCurve = metricUnitPoint(triangulation.metric, 1, [segStart, segControl, segEnd], onBoundary=True, tStart=tCurve)
            else:
                [x, y], tCurve = metricUnitPoint(triangulation.metric, 1, [segStart, segEnd], onBoundary=True, tStart=tCurve)
            t = (((x-segStart[0])**2 + (y-segStart[1])**2)/((segEnd[0]-segStart[0])**2 + (segEnd[1]-segStart[1])**2))**0.5

            # if last segment metric length less than 0.707, leave it
            if lengthInMetric(triangulation.metric, [x, y], segEnd) < 1/(2**0.5): break
            
            if t<1:
                if (int(t*10) != tP):
                    print(f"\rDiscretizing boundary; Progress: {int(t*100)}%\x1B[0K", end='')
                    tP = int(t*10)

                p1 = triangulation.meshObj['edgeSegments'][esIdx//2 + nSegs]['p1']*2
                p2 = triangulation.meshObj['edgeSegments'][esIdx//2 + nSegs]['p2']*2

                uIdx = bdySeg2Triangle(triangulation, p1, p2)
                
                bdyTriangleSubdivide(triangulation, [x, y], uIdx, triangulation.bgMetricFunction(x, y), theta=tCurve)
                nSegs += 1

                segEndP[0] = x
                segEndP[1] = y 

        return nSegs+1

# given set of points and parameter t, returns point on that curve at t
def bezier(points: List[List[float]], t: float) -> list[float]:
    """
    Returns the point on a Bézier curve at parameter t.

    This function computes a point on a Bézier curve of degree 1 or 2, depending on the number of control points.
    For degree 1, the curve is a straight line between two points, while for degree 2, it is a quadratic Bézier curve.

    Parameters:
        points (List[List[float]]): A list of control points, where each control point is a list of coordinates [x, y].
            - For degree 1, two points [start, end] are required.
            - For degree 2, three points [start, control, end] are required.
        t (float): The parameter on the curve, where t ∈ [0, 1]. t=0 corresponds to the start point, t=1 corresponds to the end point.

    Returns:
        list[float]: The [x, y] coordinates of the point on the Bézier curve at parameter t.
    """
    # bezier of degree 1
    if len(points) == 2:
        segStart = points[0]
        segEnd = points[1]
        # had to round here because when t = 1.0, result should be segEnd but due to floating point error, its a bit more than that
        return [round(segStart[0] + (segEnd[0]-segStart[0])*t, 15), round(segStart[1] + (segEnd[1]-segStart[1])*t, 15)]

    # bezier of degree 2 Rangarajan Thesis
    elif len(points) == 3:
        segStart = points[0]
        segControl = points[1]
        segEnd = points[2]
        wa = normOfVec([segStart[0]-segEnd[0], segStart[1]-segEnd[1]])
        pac = normOfVec([segStart[0]-segControl[0], segStart[1]-segControl[1]])
        pbc = normOfVec([segEnd[0]-segControl[0], segEnd[1]-segControl[1]])
        w = wa / (0.5*(pac**2 + pbc**2))**0.5
        x = segStart[0]*(1-t)**2 + segControl[0]*w*(1-t)*t + segEnd[0]*t**2
        y = segStart[1]*(1-t)**2 + segControl[1]*w*(1-t)*t + segEnd[1]*t**2

        x /= (1-t)**2 + w*(1-t)*t + t**2
        y /= (1-t)**2 + w*(1-t)*t + t**2
        return [round(x, 15), round(y, 15)]  

# returns a point along segment at a distance of given 
# metricUnit in absolute coordinates
#@TODO: current workaround for distinction when this function used for boundary discretization
# or internal point placement until iterative bdy discretization scheme is not developed.
def metricUnitPoint(
        metricFunction: Callable[[float, float], List[float]],
        metricUnit: float,
        segPoints: List[List[float]],
        iniTriangleIdx: int=0,
        onBoundary: bool=False,
        tStart: Optional[float]=None) -> List[List[float]]:
    """
    Returns a point along a segment at a given distance (metricUnit) in absolute coordinates.

    This function computes a point along a curve defined by the control points `segPoints` based on the distance 
    specified by `metricUnit`, using a root-finding approach to solve for the parameter `t` that corresponds to the 
    desired distance along the curve. The function uses the Regula Falsi method (Illinois algorithm) for root finding 
    to iteratively refine the solution. This method is used for both boundary discretization and internal point placement.

    Parameters:
        metricFunction (Callable[[float, float], List[float]]): A function that computes the metric tensor at a given point.
        metricUnit (float): The desired distance along the segment (in metric units) from the starting point.
        segPoints (List[List[float]]): A list of control points defining the segment or curve. The list should contain 
            at least two points for a line, or three points for a quadratic Bézier curve.
        iniTriangleIdx (int, optional): The index of the initial triangle for the calculation. Default is 0.
        onBoundary (bool, optional): A flag indicating whether the calculation is for a boundary discretization 
            (defaults to False).
        tStart (Optional[float], optional): The starting parameter `t` for the curve (defaults to None, 
            which will be assumed to be 0).
    
    Returns:
        List[List[float], float]: A list containing:
            - A list of coordinates [x, y] for the point at the specified distance along the segment.
            - The parameter `t` at which the point lies on the curve.

    Raises:
        ValueError: If the segment length is zero (i.e., the start and end points are the same).
        RuntimeError: If the iteration exceeds the maximum number of iterations without converging.
    """
    # length of initial given segment
    L = lengthInMetric(metricFunction, segPoints[0], segPoints[-1])

    if L <= 1e-17: raise ValueError("Selected points lead to segment of zero length")

    # segStart = segPoints[0]
    segStart = segPoints[0] if tStart == None else bezier(segPoints, tStart)
    segEnd = segPoints[-1].copy()

    # if L is less than initial segment, grow it by doubling
    # until L is longer than required metricUnit
    # if onBoundary:
    #     if L<=metricUnit:
    #         print('h', segStart, segEnd, L)
    #         while L<metricUnit:
    #             segEnd[0] = segEnd[0] + (segEnd[0]-segStart[0])*2
    #             segEnd[1] = segEnd[1] + (segEnd[1]-segStart[1])*2
    #             L = lengthInMetric(metricFunction, segStart, segEnd, iniTriangleIdx)
    # else:
    #     # if L < metricUnit:
    #     #     return segEnd
    #     if L<=metricUnit:
    #         # print('h', segStart, segEnd, L)
    #         while L<metricUnit:
    #             segEnd[0] = segEnd[0] + (segEnd[0]-segStart[0])*2
    #             segEnd[1] = segEnd[1] + (segEnd[1]-segStart[1])*2
    #             L = lengthInMetric(metricFunction, segStart, segEnd, iniTriangleIdx)
    
    # implemented regula falsi root finding to solve for
    # f(t) = lcurrent - metricUnit = 0 solving for t
    tpi = 0 if tStart == None else tStart
    tpe = 1

    # to track past values in illinois algorithm
    tpiP = None
    tpeP = None
    liP = None
    leP = None
    tpiPP = None
    tpePP = None

    f = 1
    t = None

    # iterate until f(t) < threshold
    count = 0
    # while abs(difference) > 10e-3:
    for i in range(500):
        if abs(f) < 10e-3: break
        if count == 499:raise RuntimeError(t, f, segStart, segEnd)
        
        segEndI = bezier(segPoints, tpi)
        segEndE = bezier(segPoints, tpe)

        li = lengthInMetric(metricFunction, segStart, segEndI) - metricUnit
        le = lengthInMetric(metricFunction, segStart, segEndE) - metricUnit
        
        # illinois algorithm
        if count > 0:
            # if previous 2  values are same in the row, make that function value half
            if tpi == tpiP == tpiPP:
                li = liP*0.5
            elif tpe == tpeP == tpePP:
                le = leP*0.5
        
        # finding x-intercept giving tm
        tm = (tpi*(le)-tpe*(li))/((le)-(li))

        # finding f(tm)
        segEndM = bezier(segPoints, tm)
        lm = lengthInMetric(metricFunction, segStart, segEndM) - metricUnit
        f = lm

        count+=1
        
        # track previous end points
        tpePP = tpeP
        tpeP = tpe
        tpiPP = tpiP
        tpiP = tpi
        leP = le
        liP = li

        # based on sign of difference, move a or b pivot at
        # x-intercept
        if f>0:
            tpe = tm
        else:
            tpi = tm
        t = tm

    # return [x, y] from segStart in direction of segEnd with 
    # required metricUnit length
    return bezier(segPoints, t), t

def lengthSupport(metric, segStart, segEnd):
    segStarts =[segStart]
    segEnds = [segEnd]
    metricSegStarts = [metric(segStart[0], segStart[1])]
    metricSegEnds = [metric(segEnd[0], segEnd[1])]
    l = lengthInMetricMetric(metricSegStarts[0], metricSegEnds[0], segStart, segEnd)
    lengths = [l]

    for i in range(20):
        isSegBroken = False
        # for all segments present 
        nSegs = len(segStarts)
        segIdx = 0
        for j in range(nSegs):
            # check how length is affected by adding mid point
            lOld = lengthInMetricMetric(metricSegStarts[segIdx], metricSegEnds[segIdx], segStarts[segIdx], segEnds[segIdx])

            segMiddle = [0.5*(segStarts[segIdx][0]+segEnds[segIdx][0]), 0.5*(segStarts[segIdx][1]+segEnds[segIdx][1])]
            metricSegMiddle = metric(segMiddle[0], segMiddle[1])

            lLeft = lengthInMetricMetric(metricSegStarts[segIdx], metricSegMiddle, segStarts[segIdx], segMiddle)
            lRight = lengthInMetricMetric(metricSegMiddle, metricSegEnds[segIdx], segMiddle, segEnds[segIdx])
            # ax.scatter(segMiddle[0], segMiddle[1])

            # if length varies
            if abs(lOld - (lLeft+lRight)) > 0.01/len(segStarts):
                isSegBroken = True

                localSegEnd = segEnds[segIdx]
                metricLocalSegEnd = metricSegEnds[segIdx]
                # add divided segments
                segEnds[segIdx] = segMiddle
                segStarts.insert(segIdx+1, segMiddle)
                segEnds.insert(segIdx+1, localSegEnd)

                # add metrics
                metricSegEnds[segIdx] = metricSegMiddle
                metricSegStarts.insert(segIdx+1, metricSegMiddle)
                metricSegEnds.insert(segIdx+1, metricLocalSegEnd)

                # update lengths
                lengths[segIdx] = lLeft
                lengths.insert(segIdx+1, lRight)

                segIdx += 2
            else:
                segIdx += 1
        # print(sum(lengths), len(segStarts))

        if not isSegBroken: break
        
    # generating metric function
    def edgeLength(x, y):
        geoSegStart = segStarts[0]
        geoSegEnd = segEnds[-1]

        t = (((x-geoSegStart[0])**2 + (y-geoSegStart[1])**2)/((geoSegEnd[0]-geoSegStart[0])**2 + (geoSegEnd[1]-geoSegStart[1])**2))**0.5
        segIdx = None
        for i in range(len(segEnds)):
            tEnd = (((segEnds[i][0]-geoSegStart[0])**2 + (segEnds[i][1]-geoSegStart[1])**2)/((geoSegEnd[0]-geoSegStart[0])**2 + (geoSegEnd[1]-geoSegStart[1])**2))**0.5         
            if tEnd >= t: 
                segIdx = i
                break

        # precalculated lengths
        l = sum(lengths[0:segIdx])

        # log eucledian interpolation on edge
        segStart = segStarts[segIdx]
        segEnd = segEnds[segIdx]
        # weight parameter
        tEnd = (((x-segStart[0])**2 + (y-segStart[1])**2)/((segEnd[0]-segStart[0])**2 + (segEnd[1]-segStart[1])**2))**0.5
        tStart = 1 - tEnd
        # Log Eucledian interp
        lmStart = logMetric(metricSegStarts[segIdx])
        lmEnd = logMetric(metricSegEnds[segIdx])
        # finding sum with weights
        mSum = [tStart*lmStart[0] + tEnd*lmEnd[0],
                tStart*lmStart[1] + tEnd*lmEnd[1],
                tStart*lmStart[2] + tEnd*lmEnd[2]]
        # finding metric exponential
        metricXy = expMetric(mSum)

        lm = lengthInMetricMetric(metricSegStarts[segIdx], metricXy, segStart, [x, y])

        return l+lm

    return edgeLength

def lengthInMetricGaussian(
        metricFunction: Callable[[float, float], List[float]],
        segStart: List[float],
        segEnd: List[float]) -> float:
    """
    Returns the length of a segment in a Riemannian metric space using Gaussian quadrature.

    Parameters:
        metricFunction (Callable[[float, float], List[float]]): A function that computes the metric tensor at a given point.
        segStart (list[float]): A list of two floats representing the coordinates `[x, y]` of the segment's start point.
        segEnd (list[float]): A list of two floats representing the coordinates `[x, y]` of the segment's end point.

    Returns:
        float: The length of the segment in the Riemannian metric space, computed using Gaussian quadrature.
    
    Raises:
        ValueError: If the length is computed to be less than a very small value (`EPSILON`).
    """
    EPSILON = 1e-6

    # gaussian 64 point quadrature is used for integration
    gammaPrime = [segEnd[0]-segStart[0], segEnd[1]-segStart[1]]

    L = 0
    N = GAUSSIAN_QUADR_XS_64.shape[0]
    for i in range(N):
        Mt = metricFunction(segStart[0] + 0.5*(GAUSSIAN_QUADR_XS_64[i]+1)*gammaPrime[0], segStart[1] + 0.5*(GAUSSIAN_QUADR_XS_64[i]+1)*gammaPrime[1])
        # print(Mt)
        L += GAUSSIAN_QUADR_WEIGHTS_64[i]*gammaMGamma(gammaPrime,Mt)
    L *= 0.5
    
    return L if L > EPSILON else 0

def gammaMGamma(gammaPrime: List[float], metric: List[float]) -> float:
    """
    Returns the result of the mathematical expression √(γ'ᵀ M γ'), where γ' is a vector and M is a metric.

    Parameters:
        gammaPrime (List[float]): A list of two floats representing the vector γ' (gammaPrime), 
            which contains the differences in the x and y coordinates of two points.
        metric (list[float]): A list of three floats representing the components of the metric M.

    Returns:
        float: The result of √(γ'ᵀ M γ'), which is the square root of the quadratic form computed using 
            the provided vector and metric.
    """
    # solves sqrt(gammaPrime.T M gammaPrime)
    a = metric[0]*gammaPrime[0]**2
    b = metric[1]*gammaPrime[0]*gammaPrime[1]
    c = metric[1]*gammaPrime[0]*gammaPrime[1]
    d = metric[2]*gammaPrime[1]**2
    return (a+b+c+d)**0.5

def lengthInMetricMetric(
        metricStart: List[float],
        metricEnd: List[float],
        segStart: List[float],
        segEnd: List[float]) -> float:
    """
    Computes the length of a segment in a Riemannian space, given the metric at both ends of the segment.

    Parameters:
        metricStart (List[float]): Metric at the start of the segment.
        metricEnd (list[float]): Metric at the end of the segment.
        segStart (list[float]): A list of two floats representing the coordinates of the start point of the segment.
        segEnd (list[float]): A list of two floats representing the coordinates of the end point of the segment.

    Returns:
        float: The length of the segment in the Riemannian space, considering the metric at both ends. 
              If the length is very small (below `EPSILON`), the function returns 0.
    """
    EPSILON = 1e-6

    # if segStart and segEnd are same, return 0
    if segStart == segEnd:
        return 0
    
    gammaPrime = [segEnd[0]-segStart[0], segEnd[1]-segStart[1]]

    l11 = gammaMGamma(gammaPrime, metricStart)
    l22 = gammaMGamma(gammaPrime, metricEnd)
    l1 = max(l11, l22)
    l2 = min(l11, l22)

    a = l1/l2

    # handle case of same metric at endpoints
    if a-1 < 1e-6:
        # considering liner variation on lambda (eigenval)
        L = l1 * (2/3) * (a**2 + a + 1)/(a*(a+1))
        return L
    
    L = l1 * (a-1)/(a*np.log(a))    # geometric
    # L = l1 * np.log(a)/(a-1)        # linear
    
    # # from Park, 2018
    # if abs(l11 - l22) > 0.001:
    #     L = (l11 - l22)/np.log(l11/l22)
    # else:
    #     L = 0.5*(l11 + l22)

    return L if L > EPSILON else 0

def lengthInMetric(
        metricFunction: Callable[[float, float], List[float]],
        segStart: List[float],
        segEnd: List[float]) -> float:
    """
    Computes the length of a segment in a Riemannian space, given the ends of the segment with metric.

    Parameters:
        metricFunction (Callable[[float, float], List[float]]): A function that computes the metric tensor at a given point.
        segStart (list[float]): A list of two floats representing the coordinates of the start point of the segment.
        segEnd (list[float]): A list of two floats representing the coordinates of the end point of the segment.

    Returns:
        float: The length of the segment in the Riemannian space, considering the metric at both ends. 
              If the length is very small (below `EPSILON`), the function returns 0.
    """
    EPSILON = 1e-6

    # if segStart and segEnd are same, return 0
    if segStart == segEnd:
        return 0
    
    gammaPrime = [segEnd[0]-segStart[0], segEnd[1]-segStart[1]]

    MtStart = metricFunction(segStart[0], segStart[1])
    MtEnd = metricFunction(segEnd[0], segEnd[1])

    l11 = gammaMGamma(gammaPrime, MtStart)
    l22 = gammaMGamma(gammaPrime, MtEnd)
    l1 = max(l11, l22)
    l2 = min(l11, l22)

    a = l1/l2

    # handle case of same metric at endpoints
    if a-1 < 1e-3:
        # considering liner variation on lambda (eigenval)
        L = l1 * (2/3) * (a**2 + a + 1)/(a*(a+1))
        return L
    
    L = l1 * (a-1)/(a*np.log(a))    # geometric
    # L = l1 * np.log(a)/(a-1)        # linear

    # # from Park, 2018
    # if abs(l11 - l22) > 0.001:
    #     L = (l11 - l22)/np.log(l11/l22)
    # else:
    #     L = 0.5*(l11 + l22)

    return L if L > EPSILON else 0

def angle(triangulation: 'Triangulation', nodeA: int, nodeB: int, nodeC: int) -> float:
    """
    Computes the angle subtended by vectors BA and BC in a 2D plane.

    Parameters:
        triangulation (Triangulation): An object representing a triangulated mesh.
        nodeA (int): Index of node A.
        nodeB (int): Index of node B (the vertex of the angle).
        nodeC (int): Index of node C.

    Returns:
        float: The angle (in radians) between vectors BA and BC.
    """
    coords = triangulation.coordinates
    nodeACoords = [coords[nodeA], coords[nodeA+1]]
    nodeBCoords = [coords[nodeB], coords[nodeB+1]]
    nodeCCoords = [coords[nodeC], coords[nodeC+1]]

    # finding uTV (dot product in metric)
    u = unitVec([nodeACoords[0]-nodeBCoords[0], nodeACoords[1]-nodeBCoords[1]])
    v = unitVec([nodeCCoords[0]-nodeBCoords[0], nodeCCoords[1]-nodeBCoords[1]])
    uTv = dotProduct(u, v)

    # finding theta
    angle = np.arccos(np.clip(uTv, -1, 1))
    return angle

def angleInMetric(triangulation: 'Triangulation', nodeA: int, nodeB: int, nodeC: int) -> float:
    """
    Computes the angle in a Riemannian metric space between vectors BA and BC.

    Parameters:
        triangulation (Triangulation): An object representing the triangulation, containing:
            - `coordinates`: a flat list of coordinates in [x0, y0, x1, y1, ...] form.
            - `metricMesh`: a flat list of metric tensors in [m11, m12, m22, ...] form,
              stored per node.
        nodeA (int): Index of node A.
        nodeB (int): Index of node B, at which the angle is measured.
        nodeC (int): Index of node C.

    Returns:
        float: The angle (in radians) between vectors BA and BC in the metric space.
    """
    coords = triangulation.coordinates
    nodeACoords = [coords[nodeA], coords[nodeA+1]]
    nodeBCoords = [coords[nodeB], coords[nodeB+1]]
    nodeCCoords = [coords[nodeC], coords[nodeC+1]]

    # finding metric at B
    metricAtB = [triangulation.metricMesh[int(nodeB*1.5)], triangulation.metricMesh[int(nodeB*1.5)+1], triangulation.metricMesh[int(nodeB*1.5)+2]]

    # finding uTMV (dot product in metric)
    u = [nodeACoords[0]-nodeBCoords[0], nodeACoords[1]-nodeBCoords[1]]
    v = [nodeCCoords[0]-nodeBCoords[0], nodeCCoords[1]-nodeBCoords[1]]
    uMv = u[0]*(v[0]*metricAtB[0] + v[1]*metricAtB[1]) + u[1]*(v[0]*metricAtB[1] + v[1]*metricAtB[2])

    # finding norms of u and v in metric field at B
    normU = u[0]*(u[0]*metricAtB[0] + u[1]*metricAtB[1]) + u[1]*(u[0]*metricAtB[1] + u[1]*metricAtB[2])
    normV = v[0]*(v[0]*metricAtB[0] + v[1]*metricAtB[1]) + v[1]*(v[0]*metricAtB[1] + v[1]*metricAtB[2])

    # finding theta
    angle = np.arccos(np.clip(uMv/(normU*normV)**0.5, -1, 1))
    return angle

def dotProduct(vec1: List[float], vec2: List[float]) -> float:
    """
    Computes dot product of given two vectors.

    Parameters:
        vec1 (List[float]): Vector in form of [x, y].
        vec2 (List[float]): Vector in form of [x, y].
    
    Returns:
        float: The computed dot product.
    """
    return (vec1[0]*vec2[0] + vec1[1]*vec2[1])

def normOfVec(vector: List[float]) -> float:
    """
    Computes norm of given vector.

    Parameters:
        vector (List[float]): Vector in form of [x, y]
    
    Returns:
        float: The computed vector norm (square rooted)
    """
    return (vector[0]**2 + vector[1]**2)**0.5

# returns normalized unit vector
def unitVec(vec: List[float]) -> float:
    """
    Computes normalized vector of a given vector.

    Parameters:
        vec (List[float]): Vector in form of [x, y]
    
    Returns:
        List[float]: The computed normalized vector.
    """
    return [vec[0]/normOfVec(vec), vec[1]/normOfVec(vec)]

def eigen(matrix: List[float]) -> Dict[str, float | list[float]]:
    """
    Computes the eigenvalues and eigenvectors of a 2x2 symmetric matrix.

    This function is designed for symmetric 2x2 matrices of the form:
        [ a  b ]
        [ b  c ]
    The eigenvalues and their corresponding eigenvectors are calculated analytically.
    The eigenvectors are normalized.

    Parameters:
        matrix (list[float]): A list representing the symmetric matrix [a, b, c], 
                              where:
                              - a is the (0,0) entry,
                              - b is the (0,1) and (1,0) entry,
                              - c is the (1,1) entry.

    Returns:
        dict: A dictionary with the following keys:
            - 'lambda1' (float): The larger eigenvalue.
            - 'lambda2' (float): The smaller eigenvalue.
            - 'vec1' (list[float]): The normalized eigenvector corresponding to lambda1.
            - 'vec2' (list[float]): The normalized eigenvector corresponding to lambda2.

    """
    a = matrix[0]
    b = matrix[1]
    c = matrix[2]
    
    l2 = ((a + c) - ((a - c)**2 + 4*b**2)**0.5)/2
    l1 = ((a + c) + ((a - c)**2 + 4*b**2)**0.5)/2

    # handle diagonal matrix
    if b == 0:
        return {
            'lambda1': a,       # Warning!: check robustness for a and c swapping
            'lambda2': c,
            'vec1': [1, 0],
            'vec2': [0, 1]
        }

    # v11 = -b/(b**2 + (a-l1)**2)**0.5
    # v12 = (a - l1)/(b**2 + (a-l1)**2)**0.5

    den = (b**2 + (a-l1)**2)**0.5
    if den > 1e-10:
        v11 = -b/den
        v12 = (a - l1)/den
    else:
        den = ((l1-c)**2 + b**2)**0.5
        v11 = (l1 - c)/den
        v12 = b/den

    v1 = [v11, v12]
    v2 = [-v12, v11]

    return {
        'lambda1': l1,
        'lambda2': l2,
        'vec1': v1,
        'vec2': v2
    }

# metrics
def cross(x, y):
    # x -= 0.25
    hmin = 0.005
    hmax = 0.1
    alphaX = 20*abs(x-0.5)
    alphaY = 20*abs(y-0.5)

    hx = min(2**alphaX*hmin, hmax)
    hy = min(2**alphaY*hmin, hmax)

    a = hx**(-2)
    b = 0
    c = hy**(-2)

    return [a, b, c]

def quarterCircle(x, y):
    # y-=0.5
    # x -= 0.5
    # y -= 0.5
    theta = np.arctan2(y, x)
    hmax = 0.1
    alpha = 10*abs(0.75-(x**2+y**2)**0.5)

    h1 = min(0.002*5**alpha, hmax)
    h2 = min(0.05*2**alpha, hmax)
    a = h1**(-2)*(np.cos(theta))**2 + h2**(-2)*(np.sin(theta))**2
    b = (h1**(-2) - h2**(-2))*np.cos(theta)*np.sin(theta)
    c = h1**(-2)*(np.sin(theta))**2 + h2**(-2)*(np.cos(theta))**2

    return [a, b, c]    
    
def unitMetric(x, y, unit=1):
    return [unit, 0, unit]
    
# X shape test metric from 2014 marcum paper
def xMetric(x, y):
    a = np.cosh(-100*(y-0.5-0.25*np.sin(2*np.pi*x)))**-2 #(-0.25*2*np.pi*np.cos(2*np.pi*x))
    b = np.cosh(100*(y-x))**-2

    rxx = (100*a*0.25*2*np.pi*np.cos(2*np.pi*x) - 100*b)**2
    rxy = (100*a*0.25*2*np.pi*np.cos(2*np.pi*x) - 100*b) * (-100*a + 100*b)
    ryy = (-100*a + 100*b)**2
    scale = 90
    return [(rxx+1)*scale, rxy*scale, (ryy+1)*scale]

# diagonal test metric from 2014 marcum paper
def diagonal(x, y):
    a = np.cosh(50 * (y - 0.2*x - 0.5))**-2

    rxx = 1 + (a*50*0.2)**2
    rxy = -0.2 * (50*a)**2
    ryy = 1 + (a*50)**2

    scale = 50
    return [(rxx+1)*scale, rxy*scale, (ryy+1)*scale]

# from 2022 Quasi-structured alauzet paper
def lineTest(x, y):
    h0 = 0.5
    h1 = 0.1
    h2 = 0.001
    scale = 700
    if y == 0:
        return [h1*scale, 0, h2*scale]
    else:
        return [h0*scale, 0, h0*scale]

def baryCentricCoordinates(point0: List[float], point1: List[float], point2: List[float], point: List[float]) -> List[float]:
    """
    Computes the barycentric coordinates of a given point relative to a triangle.

    Parameters:
        point0 (list[float]): Coordinates [x, y] of the first triangle vertex.
        point1 (list[float]): Coordinates [x, y] of the second triangle vertex.
        point2 (list[float]): Coordinates [x, y] of the third triangle vertex.
        point (list[float]): Coordinates [x, y] of the point for which barycentric
                             coordinates are computed.

    Returns:
        list[float]: A list of three floats representing the barycentric coordinates
                     [w1, w2, w3] corresponding to point0, point1, and point2 respectively.

    """
    # finding areas
    areaP0P1P = areab(point0, point1, point)
    areaP1P2P = areab(point1, point2, point)
    areaP2P0P = areab(point2, point0, point)
    areaTri = areab(point0, point1, point2)

    return [areaP1P2P/areaTri, areaP2P0P/areaTri, areaP0P1P/areaTri]

def areab(a: List[float], b: List[float], c: List[float]) -> float:
    """
    Computes the area of a triangle given as abc CCW points.

    Parameters:
        a (list[float]): Coordinates [x, y] of the first triangle vertex.
        b (list[float]): Coordinates [x, y] of the second triangle vertex.
        c (list[float]): Coordinates [x, y] of the third triangle vertex.

    Returns:
        float: Computed area.
    """
    return 0.5*(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))


def logMetric(metric: List[float]) -> List[float]:
    """
    Computes the logarithmic representation of a symmetric 2D metric tensor.

    The function calculates the logarithm of a 2D symmetric positive definite matrix 
    using its eigen decomposition. It returns a list representing the matrix in its 
    log-metric form [la, lb, lc], which maintains the symmetric matrix structure:

        [la, lb]
        [lb, lc]

    Parameters:
        metric (list[float]): A 3-element list representing the 2D symmetric metric tensor 
                              in the form [a, b, c] corresponding to:
                              [a, b]
                              [b, c]

    Returns:
        list[float]: A 3-element list representing the log-metric [la, lb, lc].

    """
    eigenObject = eigen(metric)
    lambda1 = eigenObject['lambda1']
    lambda2 = eigenObject['lambda2']
    vec1 = eigenObject['vec1']
    vec2 = eigenObject['vec2']
    la = (vec1[0]**2)*np.log(lambda1) + (vec1[1]**2)*np.log(lambda2)
    lb = vec1[0]*vec2[0]*np.log(lambda1) + vec1[1]*vec2[1]*np.log(lambda2)
    lc = (vec2[0]**2)*np.log(lambda1) + (vec2[1]**2)*np.log(lambda2)

    return [la, lb, lc]

def expMetric(metric: List[float]) -> List[float]:
    """
    Computes the exponential representation of a symmetric 2D metric tensor.

    This function applies the matrix exponential to a symmetric 2D matrix 
    represented in log-metric form [a, b, c] (i.e., log of a symmetric positive 
    definite matrix). It reconstructs the metric tensor by exponentiating its 
    eigenvalues and transforming it back using its eigenvectors.

    Parameters:
        metric (list[float]): A 3-element list representing the 2D symmetric metric tensor 
                              in the form [a, b, c] corresponding to:
                              [a, b]
                              [b, c]

    Returns:
        list[float]: A 3-element list representing the exponential of the input 
                     matrix [la, lb, lc], where the matrix is reconstructed as:
                     [la, lb]
                     [lb, lc]
    """
    eigenObject = eigen(metric)
    lambda1 = eigenObject['lambda1']
    lambda2 = eigenObject['lambda2']
    vec1 = eigenObject['vec1']
    vec2 = eigenObject['vec2']
    la = (vec1[0]**2)*np.exp(lambda1) + (vec1[1]**2)*np.exp(lambda2)
    lb = vec1[0]*vec2[0]*np.exp(lambda1) + vec1[1]*vec2[1]*np.exp(lambda2)
    lc = (vec2[0]**2)*np.exp(lambda1) + (vec2[1]**2)*np.exp(lambda2)

    return [la, lb, lc]
