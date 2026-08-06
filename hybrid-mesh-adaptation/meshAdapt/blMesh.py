import time
import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt
ax = plt.gca()

from .metric import unitVec, lengthInMetricMetric, gammaMGamma, logMetric, expMetric, areab
from .aflr import triangleSubdivide, localReconnection, ballOfNode, checkFlatTriangles
from .triangulationSearch import searchTriangle,isCcwOrientation, doSegmentsIntersect
from .triangulation import Triangulation
from .edgePrimitive import bdyTriangleSubdivide

# checks if given point is close to any existing point in triangulation by given distance
def isCloseToPoints(triangulation: Triangulation, point: List[float], distance: float) -> bool:
    """
    Checks if the given point is within a specified distance from any existing point 
    in the triangulation.

    Parameters:
        triangulation (Triangulation): A triangulation object that contains point coordinates and a quadtree for spatial search.
        point (List[float, float]): The point to check, represented as [x, y].
        distance (float): The distance threshold.

    Returns:
        bool: True if the point is close to any existing point in the triangulation, 
              False otherwise.
    """
    coords = triangulation.coordinates
    
    # finding nearest point
    npIdx = triangulation.quadTree.searchNearest(point[0], point[1])
    nearestPoint = [coords[npIdx], coords[npIdx+1]]
    length = ((point[0]-nearestPoint[0])**2 + (point[1]-nearestPoint[1])**2)**0.5
    if length < distance:
        # plt.scatter(nearestPoint[0], nearestPoint[1], color='g')
        return True
    
    return False

def impliedMetric(ax, ay, bx, by, cx, cy):
    e11 = bx-ax
    e12 = by-ay
    e21 = cx-bx
    e22 = cy-by
    e31 = ax-cx
    e32 = ay-cy

    det = (e11**2)*(2*e21*e22*e32**2 - 2*e31*e32*e22**2) \
        - 2*e11*e12*(e21**2 * e32**2 - e31**2 * e22**2) \
            + e12**2 * (e21**2 * 2*e31*e32 - e31**2 * 2*e21*e22)
    
    da = (2*e21*e22*e32**2 - 2*e31*e32*e22**2) \
        - (2*e11*e12)*(e32**2 - e22**2) \
            + (e12**2)*(2*e31*e32 - 2*e21*e22)
    
    db = e11**2 * (e32**2 - e22**2) \
        - (1)*(e21**2 * e32**2 - e31**2 * e22**2) \
            + e12**2 * (e21**2 - e31**2)
    
    dc = e11**2 * (2*e21*e22 - 2*e31*e32) \
        - 2*e11*e12*(e21**2 - e31**2) \
            + (1)*(e21**2 * 2*e31*e32 - e31**2 * 2*e21*e22)
    
    return [da/det, db/det, dc/det]

def blMesh(
        triangulation: Triangulation,
        bcFlag: int,
        nLayers: int,
        bgMesh: Triangulation,
        r: Optional[float]=None,
        delta0: Optional[float]=None,
        total_height: Optional[float]=None) -> tuple[List[int], List[float]]:
    """
    Generates a boundary layer mesh on a given triangulation by extending viscous 
    boundary layer points outward using smoothed averaged normals and a growth 
    rate for the layer thickness.

    Parameters:
        triangulation (Triangulation): A triangulation object containing coordinates, triangles,
                             boundary points, and additional mesh-related data.
        bcFlag (int): The boundary condition flag indicating viscous boundary segments.
        delta0 (float): Initial boundary layer thickness.
        r (float): Growth rate for successive boundary layers.
        nLayers (int): Number of boundary layer layers to generate.
        bgMesh (Triangulation): Background mesh

    Returns:
        bcFlags (list[int]): A list of flags for each triangle indicating whether it remains active (1) 
                   or was deactivated during the boundary layer growth process (0).
        newMetricMesh (list[float]): A new metricMesh for experimental blMesh-aflr transition
    """
    # console cursor controls
    UP = "\x1B[1A"
    CLR = "\x1B[0K"

    print("Boundary Layer Generation Started...")
    coords = triangulation.coordinates
    triangles = triangulation.triangles

    # list of unique boundary points; contains idxs in coords array
    boundaryPoints = list(set(triangulation.bdyPoints))

    # collect all viscous boundary layer points
    blPoints = []   # contains idxs of bl points in coords array 
    adjLeftBl = []  # contains left adj idx of bl points in coords array
    adjRightBl = [] # contains right adj idx of bl points in coords array

    avgNormals = []

    # from all boundaryPoints, collect viscous bl points based on given bcFlag
    for i in range(len(boundaryPoints)):
        
        # find segments which contain current boundaryPoint
        pointIdx = [index for index, value in enumerate(triangulation.bdyPoints) if value == boundaryPoints[i]]
        # based on index odd or even, differentiate in left or right segment
        if pointIdx[0]%2 == 0:
            rightSegIdx = pointIdx[0]//2
            leftSegIdx = (pointIdx[1]-1)//2
        else:
            leftSegIdx = (pointIdx[0]-1)//2
            rightSegIdx = pointIdx[1]//2

        # if both left and right segment bc is not viscous, skip this point
        # this can be modified to only one side being viscous if needed (for flat plate case)
        leftEdgeSeg = triangulation.meshObj['edgeSegments'][leftSegIdx]
        leftSegBc = triangulation.in2dObj['segments'][leftEdgeSeg['ednr1']]['bcFlag']
        rightEdgeSeg = triangulation.meshObj['edgeSegments'][rightSegIdx]
        rightSegBc = triangulation.in2dObj['segments'][rightEdgeSeg['ednr1']]['bcFlag']
        if leftSegBc != bcFlag and rightSegBc != bcFlag: continue

        # add viscous bl point with its left and right adjacents based on bc
        blPoints.append(boundaryPoints[i])
        if leftSegBc == bcFlag:
            adjLeftBl.append(leftEdgeSeg['p1']*2)
            xp1 = coords[leftEdgeSeg['p1']*2]
            yp1 = coords[leftEdgeSeg['p1']*2+1]
            xp2 = coords[leftEdgeSeg['p2']*2]
            yp2 = coords[leftEdgeSeg['p2']*2+1]
            leftSegNormal = unitVec([-(yp2-yp1), xp2-xp1])

            # if right segment is not bl
            if rightSegBc != bcFlag:
                # check angle between left segment normal and right segment
                xp1 = coords[rightEdgeSeg['p1']*2]
                yp1 = coords[rightEdgeSeg['p1']*2+1]
                xp2 = coords[rightEdgeSeg['p2']*2]
                yp2 = coords[rightEdgeSeg['p2']*2+1]
                rightSeg = unitVec([xp2-xp1, yp2-yp1])
                angle = np.arccos(np.clip(leftSegNormal[0]*rightSeg[0] + leftSegNormal[1]*rightSeg[1], -1, 1)) * 180/np.pi
                # if angle deviation less than 10 deg, force left seg normal towards right segment
                if angle < 10:
                    leftSegNormal = rightSeg
        else:
            adjLeftBl.append(-1)
            leftSegNormal = [0, 0]
        
        if rightSegBc == bcFlag:
            adjRightBl.append(rightEdgeSeg['p2']*2)
            xp1 = coords[rightEdgeSeg['p1']*2]
            yp1 = coords[rightEdgeSeg['p1']*2+1]
            xp2 = coords[rightEdgeSeg['p2']*2]
            yp2 = coords[rightEdgeSeg['p2']*2+1]
            rightSegNormal = unitVec([-(yp2-yp1), xp2-xp1])

            # if left segment is not bl
            if leftSegBc != bcFlag:
                # check angle between right segment normal and left segment
                xp1 = coords[leftEdgeSeg['p1']*2]
                yp1 = coords[leftEdgeSeg['p1']*2+1]
                xp2 = coords[leftEdgeSeg['p2']*2]
                yp2 = coords[leftEdgeSeg['p2']*2+1]
                leftSeg = unitVec([xp1-xp2, yp1-yp2])
                angle = np.arccos(np.clip(rightSegNormal[0]*leftSeg[0] + rightSegNormal[1]*leftSeg[1], -1, 1)) * 180/np.pi
                # if angle deviation less than 10 deg, force right seg normal towards left segment
                if angle < 10:
                    rightSegNormal = leftSeg
        else:
            adjRightBl.append(-1)
            rightSegNormal = [0, 0]

        # calculate average normal
        avgNormal = unitVec([0.5*(leftSegNormal[0]+rightSegNormal[0]), 0.5*(leftSegNormal[1] + rightSegNormal[1])])
        # plt.scatter(coords[boundaryPoints[i]] + avgNormal[0], coords[boundaryPoints[i]+1] + avgNormal[1], color='b')
        avgNormals.append(avgNormal)
    
    """
    calculate delta0 by calculating local unit metric size at each blPoint in surface 
    avg normal direction and taking the minimum of them.
    """
    minDelta0 = float('inf')
    maxDelta0 = 0
    delta0s = []
    
    if delta0 == None:
        
        # for all blPoints
        for i in range(len(blPoints)):
            # surface bl point coordinates
            xs = coords[blPoints[i]]
            ys = coords[blPoints[i]+1]
            # ax.scatter(xs, ys)

            # finding metric there
            metricBlPoint = triangulation.bgMetricFunction(xs, ys)

            # finding metric unit size in avg normal direction
            h = 1/gammaMGamma(avgNormals[i], metricBlPoint)
            delta0s.append(h)

            # if h < minDelta0:
            #     minDelta0 = h
            # if h > maxDelta0:
            #     maxDelta0 = h

            x = xs + h*(avgNormals[i][0])
            y = ys + h*(avgNormals[i][1])
            # ax.scatter(x, y)
        minDelta0 = min(delta0s)
        maxDelta0 = max(delta0s)
        avgDelta0 = sum(delta0s)/len(delta0s)
        
        t = 1.0
        delta0 = (1-t)*minDelta0 + t*avgDelta0
        # delta0 = avgDelta0
    print(minDelta0, maxDelta0, delta0)

    for i in range(len(blPoints)):
        # surface bl point coordinates
        xs = coords[blPoints[i]]
        ys = coords[blPoints[i]+1]
        # ax.scatter(xs, ys)

        x = xs + delta0*(avgNormals[i][0])
        y = ys + delta0*(avgNormals[i][1])
        # ax.scatter(x, y)
    
    # build left and right blPoint adjacents
    leftAdj = []    # contains idxs of left adj in blpoints array
    rightAdj = []   # contains idxs of right adj in blpoints array
    for i in range(len(blPoints)):
        # find index of left and right point in blPoint array
        try:
            lIdx = blPoints.index(adjLeftBl[i])
        except ValueError:
            lIdx = -1
        leftAdj.append(lIdx)
        try:
            rIdx = blPoints.index(adjRightBl[i])
        except ValueError:
            rIdx = -1
        rightAdj.append(rIdx)
    
    # for i in range(len(blPoints)):
    #     x = coords[blPoints[leftAdj[i]]]
    #     y = coords[blPoints[leftAdj[i]]+1]
    #     plt.scatter(x, y, color='r')

    # laplacian smoothing of avg normals
    # constants as given in Marcum 1995 paper
    # 1993 Pirzadeh Eq. 1
    nPasses = 10
    omega = 1
    smoothNormals = avgNormals.copy()
    smoothNormalsP = avgNormals.copy()
    # for given number of passes
    for itr in range(nPasses):
        # for all normals
        for i, normal in enumerate(smoothNormalsP):
            # if leftAdj[i] != -1:
            #     lNormalP = smoothNormalsP[leftAdj[i]]
            # else:
            #     lNormalP = [0, 0]

            # if rightAdj[i] != -1:
            #     rNormalP = smoothNormalsP[rightAdj[i]]
            # else:
            #     rNormalP = [0, 0]
            if leftAdj[i] != -1 and rightAdj[i] != -1:
                lNormalP = smoothNormalsP[leftAdj[i]]
                rNormalP = smoothNormalsP[rightAdj[i]]
                sNormalX = (1-omega)*normal[0] + omega/2 * (lNormalP[0]+rNormalP[0])
                sNormalY = (1-omega)*normal[1] + omega/2 * (lNormalP[1]+rNormalP[1])

                smoothNormals[i] = unitVec([sNormalX, sNormalY])
        
        smoothNormalsP = smoothNormals.copy()

    # for i in range(len(blPoints)):
    #     plt.scatter(coords[blPoints[i]] + smoothNormals[i][0]*0.002, coords[blPoints[i]+1] + smoothNormals[i][1]*0.002, color='r')
        # plt.scatter(coords[blPoints[i]] + avgNormals[i][0]*0.002, coords[blPoints[i]+1] + avgNormals[i][1]*0.002, color='b')

    # finding discontinuous points and adding extra normals accordingly
    normalsMapping = list(np.arange(len(blPoints))) # maps normals to bl point i.e. extra added normals are mapped back according to 
                                                    # originating blPoint (contains idxs in blPoints array) 
    # normal_discontinuity_angle = 30
    # for all blPoints
    for i in range(len(blPoints)):
        # initialize angles
        leftAngle = None
        rightAngle = None

        # if left adjacent exists as viscous segment
        if leftAdj[i] != -1:
            # left face normal
            xp1 = coords[blPoints[leftAdj[i]]]
            yp1 = coords[blPoints[leftAdj[i]]+1]
            xp2 = coords[blPoints[i]]
            yp2 = coords[blPoints[i]+1]
            leftFaceNormal = unitVec([-(yp2-yp1), xp2-xp1])

            # angle between left face normal and avg normal at blPoint
            leftAngle = np.arccos(np.clip(leftFaceNormal[0]*avgNormals[i][0] + leftFaceNormal[1]*avgNormals[i][1], -1, 1)) * 180/np.pi

            if leftAngle > 30:
                # adding extra normal found by weighted avg
                avgNormalL = unitVec([avgNormals[i][0]*2/3 + leftFaceNormal[0]/3, avgNormals[i][1]*2/3 + leftFaceNormal[1]/3])
                sNormalL = unitVec([smoothNormals[i][0]*2/3 + leftFaceNormal[0]/3, smoothNormals[i][1]*2/3 + leftFaceNormal[1]/3])
                avgNormals.append(avgNormalL)
                smoothNormals.append(sNormalL)
                normalsMapping.append(i)

        # if right adjacent exists as viscous segment
        if rightAdj[i] != -1:
            # right face normal
            xp1 = coords[blPoints[i]]
            yp1 = coords[blPoints[i]+1]
            xp2 = coords[blPoints[rightAdj[i]]]
            yp2 = coords[blPoints[rightAdj[i]]+1]
            rightFaceNormal = unitVec([-(yp2-yp1), xp2-xp1])

            # angle between right face normal and avg normal at blPoint
            rightAngle = np.arccos(np.clip(rightFaceNormal[0]*avgNormals[i][0] + rightFaceNormal[1]*avgNormals[i][1], -1, 1)) * 180/np.pi

            if rightAngle > 30:
                # adding extra normal found by weighted avg
                avgNormalR = unitVec([avgNormals[i][0]*2/3 + rightFaceNormal[0]/3, avgNormals[i][1]*2/3 + rightFaceNormal[1]/3])
                sNormalR = unitVec([smoothNormals[i][0]*2/3 + rightFaceNormal[0]/3, smoothNormals[i][1]*2/3 + rightFaceNormal[1]/3])
                avgNormals.append(avgNormalR)
                smoothNormals.append(sNormalR)
                normalsMapping.append(i)

        # delete original boundary avg normal if angle between it and extra normal is < 40 degrees
        if leftAdj[i] != -1 and leftAngle > 30:      # if extra normal exists
            if np.arccos(avgNormalL[0]*avgNormals[i][0] + avgNormalL[1]*avgNormals[i][1]) * 180/np.pi < 40:
                avgNormals[i] = -1
                smoothNormals[i] = -1
                normalsMapping[i] = -1
        # delete original boundary avg normal if angle between it and extra normal is < 40 degrees
        if rightAdj[i] != -1 and rightAngle > 30:     # if extra normal exists
            if avgNormals[i] != -1:     # if not already deleted by left face
                if np.arccos(avgNormalR[0]*avgNormals[i][0] + avgNormalR[1]*avgNormals[i][1]) * 180/np.pi < 40:
                    avgNormals[i] = -1
                    smoothNormals[i] = -1
                    normalsMapping[i] = -1

    # for i in range(len(normalsMapping)):
    #     # plt.scatter(coords[blPoints[normalsMapping[i]]] + smoothNormals[i][0]*0.002, coords[blPoints[normalsMapping[i]]+1] + smoothNormals[i][1]*0.002, color='r')
    #     if avgNormals[i] != -1:
    #         plt.scatter(coords[blPoints[normalsMapping[i]]] + avgNormals[i][0]*0.002, coords[blPoints[normalsMapping[i]]+1] + avgNormals[i][1]*0.002, color='b')


    # initializing data structure
    # for all normals proposing points
    currentBlLevel = [0]*len(avgNormals)    # initialized with zeroth layer
    lastFpoint = []                         # contains idxs of last field point placed in coords array
    for idx in normalsMapping:              # initialized with blPoints for all normals
        lastFpoint.append(blPoints[idx])
    blPointFlag = [1]*len(avgNormals)

    # for all field points
    fPointBlLevel = [-1]*(len(coords)//2)   # int of bl level for corrs field point
    fPointOrigin = [-1]*(len(coords)//2)    # contains idxs of origin points in blPoints array
    # initialization for viscous bl points
    for i in range(len(fPointOrigin)):
        try:
            index = blPoints.index(i*2)
            fPointOrigin[i] = index         # initialize with blPoints index
            fPointBlLevel[i] = 0            # initialize with 0th level
        except ValueError:
            pass
    # flags for each element; initially all active
    blFlags = [1]*(len(triangulation.triangles)//3)
    tot=0
    delta_layer = delta0
    # for given number of layers
    for itr in range(nLayers+1):        # +1 to terminate after updating flags of last layer
        # updating flag of elements
        for i in range(len(blFlags)):
            # if already off, skip this element
            if blFlags[i] == 0: continue

            p1 = triangles[i*3]
            p2 = triangles[i*3+1]
            p3 = triangles[i*3+2]

            # checking origin bl point
            p1OriginIdx = fPointOrigin[p1//2]
            p2OriginIdx = fPointOrigin[p2//2]
            p3OriginIdx = fPointOrigin[p3//2]

            if p1OriginIdx == -1 or p2OriginIdx == -1 or p3OriginIdx == -1: continue

            # identifying unique bl point origins
            blOriginsIdx = []   # manually filling as list only has 3 elements atmost
            if p1OriginIdx not in blOriginsIdx: blOriginsIdx.append(p1OriginIdx)
            if p2OriginIdx not in blOriginsIdx: blOriginsIdx.append(p2OriginIdx)
            if p3OriginIdx not in blOriginsIdx: blOriginsIdx.append(p3OriginIdx)

            if len(blOriginsIdx) > 2: continue

            lUp = 2**0.5*1.5
            lLow = 2**(-0.5)

            if len(blOriginsIdx) == 2:
                # checking if these two origin bl points are adjacents
                if blOriginsIdx[0] == leftAdj[blOriginsIdx[1]] or blOriginsIdx[0] == rightAdj[blOriginsIdx[1]]:
                    # checking if bl level difference is <= 1
                    p1Level = fPointBlLevel[p1//2]
                    p2Level = fPointBlLevel[p2//2]
                    p3Level = fPointBlLevel[p3//2]

                    if abs(p1Level-p2Level) <= 1 and abs(p2Level-p3Level) <= 1 and abs(p3Level-p1Level) <= 1:
                        blFlags[i] = 0

                        # if p1Level > 3 and p2Level > 3 and p3Level > 3:
                        #     # 11. step of Marcum 1995 paper
                        #     metricMesh = triangulation.metricMesh
                        #     m1 = [metricMesh[int(p1*1.5)], metricMesh[int(p1*1.5)+1], metricMesh[int(p1*1.5)+2]]
                        #     m2 = [metricMesh[int(p2*1.5)], metricMesh[int(p2*1.5)+1], metricMesh[int(p2*1.5)+2]]
                        #     m3 = [metricMesh[int(p3*1.5)], metricMesh[int(p3*1.5)+1], metricMesh[int(p3*1.5)+2]]
                        #     # metric length check
                        #     l1 = lengthInMetricMetric(m1, m2, [coords[p1], coords[p1+1]], [coords[p2], coords[p2+1]])
                        #     l2 = lengthInMetricMetric(m2, m3, [coords[p2], coords[p2+1]], [coords[p3], coords[p3+1]])
                        #     l3 = lengthInMetricMetric(m3, m1, [coords[p3], coords[p3+1]], [coords[p1], coords[p1+1]])
                            
                        #     if (not (lLow <= l1 <= lUp)) or (not (lLow <= l2 <= lUp)) or (not (lLow <= l3 <= lUp)):
                        #         # terminate bl generation from this blPoint
                        #         # get all avg normals from this point
                        #         for idx, value in enumerate(normalsMapping):
                        #             if value == p1OriginIdx or value == p2OriginIdx or value == p3OriginIdx:
                        #                 blPointFlag[idx] = 0
                        #                 blFlags[i] = 1

            elif len(blOriginsIdx) == 1:
                # checking if bl level difference is <= 1
                p1Level = fPointBlLevel[p1//2]
                p2Level = fPointBlLevel[p2//2]
                p3Level = fPointBlLevel[p3//2]

                if abs(p1Level-p2Level) <= 1 and abs(p2Level-p3Level) <= 1 and abs(p3Level-p1Level) <= 1:
                    blFlags[i] = 0
                    
                    # if p1Level > 3 and p2Level > 3 and p3Level > 3:
                    #     # 11. step of Marcum 1995 paper
                    #     metricMesh = triangulation.metricMesh
                    #     m1 = [metricMesh[int(p1*1.5)], metricMesh[int(p1*1.5)+1], metricMesh[int(p1*1.5)+2]]
                    #     m2 = [metricMesh[int(p2*1.5)], metricMesh[int(p2*1.5)+1], metricMesh[int(p2*1.5)+2]]
                    #     m3 = [metricMesh[int(p3*1.5)], metricMesh[int(p3*1.5)+1], metricMesh[int(p3*1.5)+2]]
                    #     # metric length check
                    #     l1 = lengthInMetricMetric(m1, m2, [coords[p1], coords[p1+1]], [coords[p2], coords[p2+1]])
                    #     l2 = lengthInMetricMetric(m2, m3, [coords[p2], coords[p2+1]], [coords[p3], coords[p3+1]])
                    #     l3 = lengthInMetricMetric(m3, m1, [coords[p3], coords[p3+1]], [coords[p1], coords[p1+1]])

                    #     if (not (lLow <= l1 <= lUp)) or (not (lLow <= l2 <= lUp)) or (not (lLow <= l3 <= lUp)):
                    #         # terminate bl generation from this blPoint
                    #         # get all avg normals from this point
                    #         for idx, value in enumerate(normalsMapping):
                    #             if value == p1OriginIdx or value == p2OriginIdx or value == p3OriginIdx:
                    #                 blPointFlag[idx] = 0
                    #                 blFlags[i] = 1
        
        if itr == nLayers: break

        # 11. step of Marcum 1995 paper not implemented yet
        
        # propose new points from all normals
        for i in range(len(avgNormals)):
            
            # check if advancement is on for this point
            if blPointFlag[i] == 0: continue
            # check if normal exists at this point
            if avgNormals[i] == -1: continue

            l = currentBlLevel[i]
            
            # delta
            if r is not None:
                delta = delta0 * (r ** l)
            else:
                delta = delta_layer

            # blending weight
            if r is not None:
                dNormalL = (1 - r**l) / (1 - r)
                dNormalN = (1 - r**nLayers) / (1 - r)
                omega = dNormalL / dNormalN
            else:
                omega = l / max(1, nLayers)
            
            # blended normal direction
            avgVecX = avgNormals[i][0] * (1 - omega) + smoothNormals[i][0] * omega
            avgVecY = avgNormals[i][1] * (1 - omega) + smoothNormals[i][1] * omega
            avgVec = unitVec([avgVecX, avgVecY])

            x = coords[lastFpoint[i]] + avgVec[0]*delta
            y = coords[lastFpoint[i]+1] + avgVec[1]*delta
            # plt.scatter(x, y, color='r')
            
            # 13. step of Marcum 1995 paper not implemented yet

            # if is close to existing triangulation point
            """
            taking min of local normal and tangential spacing
            """
            leftPoint = [coords[blPoints[leftAdj[normalsMapping[i]]]], coords[blPoints[leftAdj[normalsMapping[i]]]+1]]
            rightPoint = [coords[blPoints[rightAdj[normalsMapping[i]]]], coords[blPoints[rightAdj[normalsMapping[i]]]+1]]
            centrePoint = [coords[blPoints[normalsMapping[i]]], coords[blPoints[normalsMapping[i]]+1]]
            
            tauLeft = ((centrePoint[0]-leftPoint[0])**2 + (centrePoint[1]-leftPoint[1])**2)**0.5
            tauRight = ((centrePoint[0]-rightPoint[0])**2 + (centrePoint[1]-rightPoint[1])**2)**0.5

            if isCloseToPoints(triangulation, [x, y], min(delta, tauLeft, tauRight)*0.7):
                # plt.scatter(x, y, color='r')
                # plt.scatter(leftPoint[0], leftPoint[1])
                # terminate futher advancement
                blPointFlag[i] = 0
                continue
            
            triIdx = searchTriangle(x, y, triangulation)
            if type(triIdx) == list:
                triIdx = triIdx[0]
            
            # if point on edge, consider both adjacent triangles
            if type(triIdx) == list:
                # if any one triangle is already turned off
                if blFlags[triIdx[0]//3] == 0 or blFlags[triIdx[1]//3] == 0:
                    # terminate futher advancement
                    blPointFlag[i] = 0
                    continue
                
                # for both triangles
                for tri in triIdx:
                    p1 = triangles[tri]
                    p2 = triangles[tri+1]
                    p3 = triangles[tri+2]

                    # checking origin bl point for all three points
                    p1OriginIdx = fPointOrigin[p1//2]
                    p2OriginIdx = fPointOrigin[p2//2]
                    p3OriginIdx = fPointOrigin[p3//2]

                    # finding allowed origins based on current blPoint and its adjacents
                    p0 = normalsMapping[i]
                    pL = leftAdj[normalsMapping[i]]
                    pR = rightAdj[normalsMapping[i]]
                    allowedOrigins = [-1, p0, pL, pR]

                    # if triangle contains points of which origins are not in allowedOrigins array
                    if p1OriginIdx not in allowedOrigins or p2OriginIdx not in allowedOrigins or p3OriginIdx not in allowedOrigins:
                        # terminate futher advancement
                        blPointFlag[i] = 0
                        continue
            
            else:
                if blFlags[triIdx//3] == 0:
                    # terminate futher advancement
                    blPointFlag[i] = 0
                    continue

                p1 = triangles[triIdx]
                p2 = triangles[triIdx+1]
                p3 = triangles[triIdx+2]

                # checking origin bl point for all three points
                p1OriginIdx = fPointOrigin[p1//2]
                p2OriginIdx = fPointOrigin[p2//2]
                p3OriginIdx = fPointOrigin[p3//2]

                # finding allowed origins based on current blPoint and its adjacents
                p0 = normalsMapping[i]
                pL = leftAdj[normalsMapping[i]]
                pR = rightAdj[normalsMapping[i]]
                allowedOrigins = [-1, p0, pL, pR]

                # if triangle contains points of which origins are not in allowedOrigins array
                # if p1OriginIdx not in allowedOrigins or p2OriginIdx not in allowedOrigins or p3OriginIdx not in allowedOrigins:
                #     # terminate futher advancement
                #     blPointFlag[i] = 0
                #     continue
            # plt.scatter(x, y, color='r')

            # checking if point on boundary
            if -1 in triangulation.adjacents[triIdx:triIdx+3]:
                isSubdivided = False
                # for all edges of boundary triangle
                for bdyIdx in range(3):
                    # bdyIdx = triangulation.adjacents[triIdx:triIdx+3].index(-1)
                    ux = coords[triangles[triIdx+bdyIdx]]
                    uy = coords[triangles[triIdx+bdyIdx]+1]
                    vx = coords[triangles[triIdx+(bdyIdx+1)%3]]
                    vy = coords[triangles[triIdx+(bdyIdx+1)%3]+1]
                    ox = coords[triangles[triIdx+(bdyIdx+2)%3]]
                    oy = coords[triangles[triIdx+(bdyIdx+2)%3]+1]
                    # if point is on the current edge 
                    if isCcwOrientation(ux, uy, vx, vy, x, y) == 0 and doSegmentsIntersect(ux, uy, vx, vy, ox, oy, x, y):
                        # if current edge is on the boundary
                        if triangulation.adjacents[triIdx+bdyIdx] == -1:
                            # subdivide bdy triangle
                            uIdx = triIdx + bdyIdx
                            bdyTriangleSubdivide(triangulation, [x, y], uIdx, triangulation.bgMetricFunction(x, y))
                            blFlags.append(1)
                            isSubdivided = True
                        else:
                            # subdivide proposed point
                            triangleSubdivide(triangulation, [x, y], triIdx, triangulation.bgMetricFunction(x, y))
                            blFlags.append(1)
                            blFlags.append(1)
                            isSubdivided = True
                
                # if point is inside boundary triangle, not on any edges
                if not isSubdivided:
                    # subdivide proposed point
                    triangleSubdivide(triangulation, [x, y], triIdx, triangulation.bgMetricFunction(x, y))
                    blFlags.append(1)
                    blFlags.append(1)
            else:
                # subdivide proposed point
                triangleSubdivide(triangulation, [x, y], triIdx, triangulation.bgMetricFunction(x, y))
                blFlags.append(1)
                blFlags.append(1)

            # update data structure
            currentBlLevel[i] += 1
            lastFpoint[i] = len(coords)-2
            fPointBlLevel.append(itr)
            fPointOrigin.append(p0)
            # blFlags.append(1)
            # blFlags.append(1)
        st=time.time()
        # local reconnect for current layer using min-max criterion based on physical angles
        swaps = localReconnection(triangulation, criterion="min-max", skipTriangleFlags=blFlags)
        tot+=time.time()-st
        print(f"\rIteration: {itr+1}{CLR}", end='')
        
        if r is None and itr < nLayers:
            deltaVals = []

            for i in range(len(avgNormals)):
                if blPointFlag[i] == 0:
                    continue
                if avgNormals[i] == -1:
                    continue

                xf = coords[lastFpoint[i]]
                yf = coords[lastFpoint[i] + 1]

                metricF = triangulation.bgMetricFunction(xf, yf)
                nvec = avgNormals[i]

                h = 1.0 / gammaMGamma(nvec, metricF)
                deltaVals.append(h)

            if len(deltaVals) > 0:
                minDelta = min(deltaVals)
                maxDelta = max(deltaVals)
                avgDelta = sum(deltaVals) / len(deltaVals)
                t = 1.0
                delta_layer = (1 - t) * minDelta + t * avgDelta
                # delta_layer = avgDelta
                print(f"[Layer {itr+1}] delta_layer:", minDelta, maxDelta, delta_layer)
                
        # print(f"\rIteration: {itr+1}{CLR}: ", end='')

    print(f"\rBoundary layer generation completed with total iterations: {itr+1}{CLR}")
    print("Total time taken for reconnection:", tot)
    
    # set implied metric
    newMetricMesh = bgMesh.metricMesh.copy()
    # for each last point
    for i in range(len(lastFpoint)):
        point = lastFpoint[i]

        # finding ball of node around point
        triAroundPoint = ballOfNode(triangulation, point)

        mSum = [0, 0, 0]
        areas = 0

        for triIdx in triAroundPoint:
            # if triangle is off
            if blFlags[triIdx//3] == 0:
                ax = coords[triangles[triIdx]]
                ay = coords[triangles[triIdx]+1]
                bx = coords[triangles[triIdx+1]]
                by = coords[triangles[triIdx+1]+1]
                cx = coords[triangles[triIdx+2]]
                cy = coords[triangles[triIdx+2]+1]

                # implied metric of triangle
                iMetric = impliedMetric(ax, ay, bx, by, cx, cy)

                # ln metric
                lMetric = logMetric(iMetric)

                # area of triangle
                area = areab([ax, ay], [bx, by], [cx, cy])

                mSum[0] += area*lMetric[0]
                mSum[1] += area*lMetric[1]
                mSum[2] += area*lMetric[2]
                areas += area
        
        if areas == 0: continue

        mSum[0] /= areas
        mSum[1] /= areas
        mSum[2] /= areas

        # exponential metric
        mP = expMetric(mSum)

        # find nearest point in bgMesh
        nPoint = bgMesh.quadTree.searchNearest(coords[point], coords[point+1])

        # set new metric as the implied one
        newMetricMesh[int(nPoint*1.5)] = mP[0]
        newMetricMesh[int(nPoint*1.5)+1] = mP[1]
        newMetricMesh[int(nPoint*1.5)+2] = mP[2]
        
    print("Inside blMesh")
        
    checkFlatTriangles(triangulation.coordinates, triangulation.triangles)
    
    return blFlags, newMetricMesh
