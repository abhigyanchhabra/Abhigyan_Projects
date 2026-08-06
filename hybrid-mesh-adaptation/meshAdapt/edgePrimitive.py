from meshAdapt import exportMesh

from .metric import lengthInMetricMetric, lengthInMetric, gammaMGamma, expMetric, areab, bezier
from .triangulationSearch import collectEdges, ballOfNode
from .aflr import triangleSubdivide
from .elementADT import buildElementADT, elementBoundingBox
from .triangulation import Triangulation
from matplotlib import pyplot as plt
from typing import List, Tuple

# ax = plt.gca()
upper_limit = 1.414
lower_limit = 0.7071

def heapSort(original):
    n = len(original)
    sortedIdxs = list(range(n))

    # trivial case
    if n < 2: return sortedIdxs

    # initializing heap construction parameters
    l = (n//2) + 1
    ir = n-1

    while True:
        # first phase; building heap
        if l > 1:
            l -= 1
            idxt = sortedIdxs[l-1]
            q = original[idxt]
        # second phase; extracting elements from heap
        else:
            idxt = sortedIdxs[ir]
            q = original[idxt]
            sortedIdxs[ir] = sortedIdxs[0]  # move root to end
            ir -= 1

            # if processed all elements
            if ir == 0:
                sortedIdxs[0] = idxt
                break

        # sift down to restore heap property
        i = l-1
        j = l+i

        while j <= ir:
            # if two children, choose larger one
            if j < ir and original[sortedIdxs[j]] < original[sortedIdxs[j+1]]:
                j += 1
            
            # if current element is smaller than larger child, move child up
            if q < original[sortedIdxs[j]]:
                sortedIdxs[i] = sortedIdxs[j]
                i = j
                # next child position
                j = 2*j + 1
            else:
                # heap property satisfied
                break 

        # place current element in its correct position
        sortedIdxs[i] = idxt

    return sortedIdxs

def classifyPoints(triangulation, draw=None):
    EPSILON = 1e-15
    # initialize all points as 0 which represents internal points
    pointLevel = [0]*(len(triangulation.coordinates)//2)

    # loop trhough all edge segments in meshobj
    edgen = len(triangulation.meshObj['edgeSegments'])
    for i in range(edgen):
        edgeSeg = triangulation.meshObj['edgeSegments'][i]
        p1 = edgeSeg['p1']
        p2 = edgeSeg['p2']
        # ax.scatter(triangulation.coordinates[p1*2], triangulation.coordinates[p1*2+1])
        # ax.scatter(triangulation.coordinates[p2*2], triangulation.coordinates[p2*2+1])

        # if p1 is singular point
        # TODO: workaround currently for reentry problem singular points
        if abs(edgeSeg['dist1']-0) < EPSILON:
            pointLevel[p1] = 2
        else:
            pointLevel[p1] = 1
            # if draw:
            #     ax.scatter(triangulation.coordinates[p1*2], triangulation.coordinates[p1*2+1])

        # if p2 is singular point
        # TODO: workaround currently for reentry problem singular points
        if abs(edgeSeg['dist2']-1) < EPSILON:
            pointLevel[p2] = 2
        else:
            pointLevel[p2] = 1
            # if draw:
            #     ax.scatter(triangulation.coordinates[p2*2], triangulation.coordinates[p2*2+1])
    
    return pointLevel

def findEdgeSegment(triangulation, p1, p2=None):
    # edit by Abhigyan: check both p1->p2 and p2->p1 directions so edges added by
    # insert_quad_geometry (whose storage direction may differ from the triangulation's
    # traversal order) are found correctly
    if p2 is not None:
        n1, n2 = p1 // 2, p2 // 2  # plain node indices
        edgen = len(triangulation.meshObj['edgeSegments'])
        for i in range(edgen):
            edgeSeg = triangulation.meshObj['edgeSegments'][i]
            if (edgeSeg['p1'] == n1 and edgeSeg['p2'] == n2) or \
               (edgeSeg['p1'] == n2 and edgeSeg['p2'] == n1):
                return i
    else:
        edgen = len(triangulation.meshObj['edgeSegments'])
        for i in range(edgen):
            edgeSeg = triangulation.meshObj['edgeSegments'][i]
            if edgeSeg['p1'] == p1 // 2:
                return i

    raise RuntimeError

def bdyEdgeSubdivide(triangulation, point, esIdx, theta=None):
    bdyPoints = triangulation.bdyPoints
    coords = triangulation.coordinates

    esIdx *= 2

    # finding geometry segment for curved-boundary support
    geoSegIdx = triangulation.meshObj['edgeSegments'][esIdx//2]['ednr1']
    geoSeg = triangulation.in2dObj['segments'][geoSegIdx]
    in2dPoints = triangulation.in2dObj['points']

    # determine whether this mesh edge sits on a curved (3-point Bézier) geometric segment
    if 'p3' in geoSeg:
        gp1 = geoSeg['p1'] * 2
        gpc = geoSeg['p2'] * 2   # control point index in in2d points array
        gp2 = geoSeg['p3'] * 2
        geoSegStart   = [in2dPoints[gp1],  in2dPoints[gp1 + 1]]
        geoSegControl = [in2dPoints[gpc],  in2dPoints[gpc + 1]]
        geoSegEnd     = [in2dPoints[gp2],  in2dPoints[gp2 + 1]]
        isCurved = True
    else:
        isCurved = False

    theta1 = triangulation.meshObj['edgeSegments'][esIdx//2]['dist1']
    theta2 = triangulation.meshObj['edgeSegments'][esIdx//2]['dist2']

    segStart = [coords[bdyPoints[esIdx]], coords[bdyPoints[esIdx]+1]]
    segEnd = [coords[bdyPoints[esIdx+1]], coords[bdyPoints[esIdx+1]+1]]
    
    # remove edgeSegment 
    segmentObjs = [triangulation.meshObj['edgeSegments'].pop(esIdx//2)]

    # if theta not supplied, compute it from the linear position along the mesh edge,
    # then remap into the [theta1, theta2] range on the geometric edge
    if theta is None:
        # find normalised arc-length parameter along the straight mesh edge
        dx = segEnd[0] - segStart[0]
        dy = segEnd[1] - segStart[1]
        seg_len_sq = dx*dx + dy*dy
        t = (((point[0]-segStart[0])**2 + (point[1]-segStart[1])**2) / seg_len_sq)**0.5

        # linearly rescale parameter to get parameter on geometric edge
        # [0, t, 1] -> [theta1, theta, theta2]
        theta = theta1 + t*(theta2-theta1)

    # if the geometric edge is curved, snap the new node exactly onto the Bézier curve
    # and update the stored coordinates and metrics accordingly
    if isCurved:
        x, y = bezier([geoSegStart, geoSegControl, geoSegEnd], theta)
        # overwrite the coordinates that were appended by bdyTriangleSubdivide
        coords[-2] = x
        coords[-1] = y
        # recompute and overwrite the metric at the snapped position
        if triangulation.bgMetricFunction:
            snappedMetric = triangulation.bgMetricFunction(x, y)
        else:
            snappedMetric = triangulation.metric(x, y)
        snappedMetricLog = logMetric(snappedMetric)
        triangulation.metricMesh[-3] = snappedMetric[0]
        triangulation.metricMesh[-2] = snappedMetric[1]
        triangulation.metricMesh[-1] = snappedMetric[2]
        triangulation.metricLog[-3] = snappedMetricLog[0]
        triangulation.metricLog[-2] = snappedMetricLog[1]
        triangulation.metricLog[-1] = snappedMetricLog[2]

    # build new segment object
    segObj = {
        'surfid': segmentObjs[-1]['surfid'],
        'p1': len(coords)//2-1,
        'p2': segmentObjs[-1]['p2'],
        'sf1': segmentObjs[-1]['sf1'],
        'sf2': segmentObjs[-1]['sf2'],
        'ednr1': segmentObjs[-1]['ednr1'],
        'dist1': theta,
        'ednr2': segmentObjs[-1]['ednr2'],
        'dist2': segmentObjs[-1]['dist2']
    }

    # edit previous segment object
    segmentObjs[-1]['p2'] = len(coords)//2-1
    segmentObjs[-1]['dist2'] = theta

    # append new segment object
    segmentObjs.append(segObj)

    # adding back all edge segments
    triangulation.meshObj['edgeSegments'][esIdx//2:esIdx//2] = segmentObjs
    bdyPoints[esIdx+1:esIdx+1] = [len(coords)-2, len(coords)-2]

# alters boundary triangle by inserting given point and subdividing it 
# TODO: merge this with triangle subdivide
from meshAdapt import logMetric
def bdyTriangleSubdivide(triangulation: Triangulation, point, uIdx, pointMetric=None, theta=None):
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

    # storing triangle structure before subdivision
    triangleIndex = uIdx - uIdx%3
    uNode = triangulation.triangles[uIdx]
    vNode = triangulation.triangles[triangleIndex + (uIdx+1)%3]
    aNode = triangulation.triangles[triangleIndex + (uIdx+2)%3]
    auAdj = triangulation.adjacents[triangleIndex + (uIdx+2)%3]

    # shifting u to newNode
    triangulation.triangles[uIdx] = newNode

    # adding new triangle along au
    triangulation.triangles.append(aNode)
    triangulation.triangles.append(uNode)
    triangulation.triangles.append(newNode)

    # changing adjacents of old triangle
    triangulation.adjacents[triangleIndex + (uIdx+2)%3] = len(triangulation.triangles)-3

    # adding adjacents of au triangle
    triangulation.adjacents.append(auAdj)
    triangulation.adjacents.append(-1)
    triangulation.adjacents.append(triangleIndex)

    # for auAdj triangle, setting new au triangle as adjacent
    if auAdj != -1:
        for i in range(auAdj, auAdj+3):
            if triangulation.adjacents[i] == triangleIndex:
                triangulation.adjacents[i] = len(triangulation.triangles)-3
                
    # blflags: appending flags for BL elements
    if triangulation.blFlags:
        oldTriIdx = triangleIndex // 3
        oldBLFlag = triangulation.blFlags[oldTriIdx]

        # original triangle keeps same BL flag
        triangulation.blFlags[oldTriIdx] = oldBLFlag

        # new triangle inherits BL flag
        triangulation.blFlags.append(oldBLFlag)

    if triangulation.flags:
        # turning on flags of old triangles
        triangulation.flags[int(triangleIndex/3)] = 1

        # adding flag for newly created triangle
        triangulation.flags.append(1)

    # insert new point to quadtree and attach new triangle here
    triangulation.quadTree.insert(len(triangulation.coordinates)-2)
    triangulation.adjPointTris.append(len(triangulation.triangles)-3)

    # change old triangle attached to u point
    if triangulation.adjPointTris[uNode//2] == triangleIndex:
        triangulation.adjPointTris[uNode//2] = len(triangulation.triangles)-3

    # incrementally update element ADT: the subdivided triangle changed
    # bounding box, and 1 new triangle was appended to the end
    triangulation.elementADT.update(triangleIndex, elementBoundingBox(triangulation, triangleIndex))
    nTriangles = len(triangulation.triangles)
    triangulation.elementADT.insert(elementBoundingBox(triangulation, nTriangles-3), nTriangles-3)

    # divide boundary edge
    edgeSegIdx = findEdgeSegment(triangulation, uNode, vNode)
    bdyEdgeSubdivide(triangulation, point, edgeSegIdx, theta=theta)

def nodesAroundNode(triangulation, node):
    triangles = triangulation.triangles
    adjacents = triangulation.adjacents

    # seed triangle to start with
    seedTriIdx = triangulation.adjPointTris[node//2]
    currentTriIdx = seedTriIdx

    nodes = []

    forwardPass = True

    for i in range(500):
        if i == 499: raise ValueError
        
        # find local index of node in triangle
        nodeIdx = [triangles[currentTriIdx], triangles[currentTriIdx+1], triangles[currentTriIdx+2]].index(node)
        if forwardPass:
            # finding forward adjacent
            adjTriIdx = adjacents[currentTriIdx+(nodeIdx-1)%3]
            nodes.append(triangles[currentTriIdx+(nodeIdx-1)%3])
        else:
            # finding backward adjacent
            adjTriIdx = adjacents[currentTriIdx+(nodeIdx)%3]
            nodes.append(triangles[currentTriIdx+(nodeIdx+1)%3])
        
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

        # move to adjacent triangle
        currentTriIdx = adjTriIdx
    
    if len(nodes) != len(list(set(nodes))):
        raise ValueError
    return nodes

def edgesAroundNode(triangulation, node):
    triangles = triangulation.triangles
    adjacents = triangulation.adjacents

    # seed triangle to start with
    seedTriIdx = triangulation.adjPointTris[node//2]
    currentTriIdx = seedTriIdx

    edges = []

    forwardPass = True

    for i in range(500):
        if i == 499: raise ValueError

        # find local index of node in triangle

        nodeIdx = [triangles[currentTriIdx], triangles[currentTriIdx+1], triangles[currentTriIdx+2]].index(node)
        if forwardPass:
            # finding forward adjacent
            adjTriIdx = adjacents[currentTriIdx+(nodeIdx+2)%3]
            edges.append(currentTriIdx+(nodeIdx+2)%3)
        else:
            # finding backward adjacent
            adjTriIdx = adjacents[currentTriIdx+(nodeIdx)%3]
            edges.append(currentTriIdx+(nodeIdx)%3)
        
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

        # move to adjacent triangle
        currentTriIdx = adjTriIdx
    
    if len(edges) != len(list(set(edges))):
        raise ValueError
    return edges

def edgeMetricInterpolate(triangulation, p1, p2, point):
    coords = triangulation.coordinates
    metricLog = triangulation.metricLog

    segStart = [coords[p1], coords[p1+1]]
    segEnd = [coords[p2], coords[p2+1]]

    # weight parameter
    tEnd = (((point[0]-segStart[0])**2 + (point[1]-segStart[1])**2)/((segEnd[0]-segStart[0])**2 + (segEnd[1]-segStart[1])**2))**0.5
    tStart = 1 - tEnd

    # Log Eucledian interp
    lmStart = [metricLog[int(p1*1.5)], metricLog[int(p1*1.5)+1], metricLog[int(p1*1.5)+2]]
    lmEnd = [metricLog[int(p2*1.5)], metricLog[int(p2*1.5)+1], metricLog[int(p2*1.5)+2]]

    # finding sum with weights
    mSum = [tStart*lmStart[0] + tEnd*lmEnd[0],
            tStart*lmStart[1] + tEnd*lmEnd[1],
            tStart*lmStart[2] + tEnd*lmEnd[2]]
    
    # finding metric exponential
    mX = expMetric(mSum)

    return mX

def lengthInNodeMetric(nodeMetric, segStart, segEnd):
    # if segStart and segEnd are same, return 0
    if segStart == segEnd:
        return 0
    
    gammaPrime = [segEnd[0]-segStart[0], segEnd[1]-segStart[1]]

    l = gammaMGamma(gammaPrime, nodeMetric)

    return l

def splitEdge(triangulation, uIdx):
    coords = triangulation.coordinates

    aTriStartIdx = uIdx - uIdx%3
    p1 = triangulation.triangles[uIdx]
    p2 = triangulation.triangles[aTriStartIdx + (uIdx+1)%3]

    segStart = [coords[p1], coords[p1+1]]
    segEnd = [coords[p2], coords[p2+1]]

    metricP1 = [triangulation.metricMesh[int(p1*1.5)], triangulation.metricMesh[int(p1*1.5)+1], triangulation.metricMesh[int(p1*1.5)+2]]
    metricP2 = [triangulation.metricMesh[int(p2*1.5)], triangulation.metricMesh[int(p2*1.5)+1], triangulation.metricMesh[int(p2*1.5)+2]]

    # finding lengths in endpoint metrics
    l1 = lengthInNodeMetric(metricP1, segStart, segEnd)
    l2 = lengthInNodeMetric(metricP2, segStart, segEnd)

    # finding length in metric field
    l12 = lengthInMetricMetric(metricP1, metricP2, segStart, segEnd)
    
    # finding weight of node 2 for interpolation of new point
    node2Wt = 0.5
    t = l1/(l1+l2)
    if 0.25 < t < 0.75:
        node2Wt = 1 - t
    else:
        if l1 < l2:
            node2Wt = 1 - l1/l12
        else:
            node2Wt = l2/l12
    
    # clipping node weight
    node2Wt = min(0.95, max(0.05, node2Wt))

    # finding new node
    x = (1-node2Wt)*segStart[0] + node2Wt*segEnd[0]
    y = (1-node2Wt)*segStart[1] + node2Wt*segEnd[1]

    # check if point is on boundary edge
    if triangulation.adjacents[uIdx] == -1:
        try:
            edgeSegIdx = findEdgeSegment(triangulation, p1, p2)
        except RuntimeError:
            # edit by Abhigyan: boundary edge has no edgeSegments record (e.g. newly inserted
            # geometry edges that weren't in the original .vol); treat as straight line
            edgeSegIdx = None
        if edgeSegIdx is not None:
            geoSegIdx = triangulation.meshObj['edgeSegments'][edgeSegIdx]['ednr1']
            # check if point is on curved geometric segment
            if 'p3' in triangulation.in2dObj['segments'][geoSegIdx]:
                # use separate geo_* variables to avoid shadowing the mesh-node p1/p2
                geo_p1 = triangulation.in2dObj['segments'][geoSegIdx]['p1'] * 2
                geo_pc = triangulation.in2dObj['segments'][geoSegIdx]['p2'] * 2   # control point
                geo_p2 = triangulation.in2dObj['segments'][geoSegIdx]['p3'] * 2

                in2dPoints = triangulation.in2dObj['points']

                geoSegStart   = [in2dPoints[geo_p1],  in2dPoints[geo_p1 + 1]]
                geoSegControl = [in2dPoints[geo_pc],  in2dPoints[geo_pc + 1]]   # always present for 3-pt seg
                geoSegEnd     = [in2dPoints[geo_p2],  in2dPoints[geo_p2 + 1]]

                theta1 = triangulation.meshObj['edgeSegments'][edgeSegIdx]['dist1']
                theta2 = triangulation.meshObj['edgeSegments'][edgeSegIdx]['dist2']

                # linearly rescale parameter to get parameter on geometric edge
                # [0, node2Wt, 1] -> [theta1, theta, theta2]
                theta = theta1 + (1-node2Wt)*(theta2-theta1)
                x, y = bezier([geoSegStart, geoSegControl, geoSegEnd], theta)

    # if bg metric function available, use that
    # NOTE: p1/p2 are still the original mesh-node coordinate indices here
    if triangulation.bgMetricFunction:
        newMetric = triangulation.bgMetricFunction(x, y)
    else:
        newMetric = edgeMetricInterpolate(triangulation, p1, p2, [x, y])

    return (x, y), newMetric

def det(metric):
    return metric[0]*metric[2] - metric[1]*metric[1]

def splitTriQuality(triangulation, uIdx, newPoint, newMetric):
    coords = triangulation.coordinates
    triangles = triangulation.triangles
    adjacents = triangulation.adjacents
    metricMesh = triangulation.metricMesh

    # finding two triangles adjacent to the given edge
    aTriStartIdx = uIdx - uIdx%3
    bTriStartIdx = adjacents[uIdx]
    uNode = triangulation.triangles[uIdx]
    vNode = triangulation.triangles[aTriStartIdx + (uIdx%3+1)%3]

    # finding a node
    aNodeIdx = aTriStartIdx + ((uIdx%3) + 2)%3
    aNode = triangulation.triangles[aNodeIdx]
    
    # metric at all nodes
    mU = [metricMesh[int(uNode*1.5)], metricMesh[int(uNode*1.5)+1], metricMesh[int(uNode*1.5)+2]]
    mV = [metricMesh[int(vNode*1.5)], metricMesh[int(vNode*1.5)+1], metricMesh[int(vNode*1.5)+2]]
    mA = [metricMesh[int(aNode*1.5)], metricMesh[int(aNode*1.5)+1], metricMesh[int(aNode*1.5)+2]]
    
    # determinants
    detMU = det(mU)
    detMV = det(mV)
    detMA = det(mA)
    detMNew = det(newMetric)

    # quality for the anv triangle
    a0Area = areab([coords[aNode], coords[aNode+1]], newPoint, [coords[vNode], coords[vNode+1]])
    minDet = min(detMA, detMNew, detMV)
    lan = lengthInMetricMetric(mA, newMetric, [coords[aNode], coords[aNode+1]], newPoint)
    lnv = lengthInMetricMetric(newMetric, mV, newPoint, [coords[vNode], coords[vNode+1]])
    lva = lengthInMetricMetric(mV, mA, [coords[vNode], coords[vNode+1]], [coords[aNode], coords[aNode+1]])
    qAnv = 4/3**0.5 * 3 * a0Area * minDet**0.5 / (lan*lan + lnv*lnv + lva*lva)

    # quality for aun triangle
    a1Area = areab([coords[aNode], coords[aNode+1]], [coords[uNode], coords[uNode+1]], newPoint)
    minDet = min(detMA, detMU, detMNew)
    lau = lengthInMetricMetric(mA, mU, [coords[aNode], coords[aNode+1]], [coords[uNode], coords[uNode+1]])
    lun = lengthInMetricMetric(mU, newMetric, [coords[uNode], coords[uNode+1]], newPoint)
    qAun = 4/3**0.5 * 3 * a1Area  * minDet**0.5 / (lau*lau + lun*lun + lan*lan)

    # if there exist b adjacent triangle
    if bTriStartIdx != -1:
        # finding b node in adjacent triangle
        # TODO:
        if bTriStartIdx == -1: return -1
        if triangles[bTriStartIdx] == uNode:
            bNode = triangles[bTriStartIdx+1]
            bNodeIdx = bTriStartIdx+1
        elif triangles[bTriStartIdx+1] == uNode:
            bNode = triangles[bTriStartIdx+2]
            bNodeIdx = bTriStartIdx+2
        elif triangles[bTriStartIdx+2] == uNode:
            bNode = triangles[bTriStartIdx]
            bNodeIdx = bTriStartIdx
        else:
            print(bTriStartIdx, triangles[bTriStartIdx] , triangles[bTriStartIdx+1], triangles[bTriStartIdx+2])
            raise RuntimeError
        
        mB = [metricMesh[int(bNode*1.5)], metricMesh[int(bNode*1.5)+1], metricMesh[int(bNode*1.5)+2]]
        detMB = det(mB)

        # quality for bvn triangle
        b0Area = areab([coords[bNode], coords[bNode+1]], [coords[vNode], coords[vNode+1]], newPoint)
        minDet = min(detMB, detMV, detMNew)
        lbv = lengthInMetricMetric(mB, mV, [coords[bNode], coords[bNode+1]], [coords[vNode], coords[vNode+1]])
        lnb = lengthInMetricMetric(newMetric, mB, newPoint, [coords[bNode], coords[bNode+1]])
        qBvn = 4/3**0.5 * 3 * b0Area * minDet**0.5 / (lbv*lbv + lnv*lnv + lnb*lnb)

        # quality for bnu triangle
        b1Area = areab([coords[bNode], coords[bNode+1]], newPoint, [coords[uNode], coords[uNode+1]])
        minDet = min(detMB, detMNew, detMU)
        lub = lengthInMetricMetric(mU, mB, [coords[uNode], coords[uNode+1]], [coords[bNode], coords[bNode+1]])
        qBnu = 4/3**0.5 * 3 * b1Area * minDet**0.5 / (lnb*lnb + lun*lun + lub*lub)

        return min(qAnv, qAun, qBvn, qBnu)
    
    return min(qAnv, qAun)

def splitPass(triangulation: Triangulation):
    coords = triangulation.coordinates
    adjacents = triangulation.adjacents

    # master list of edges
    edges = collectEdges(triangulation)

    pointLevels = classifyPoints(triangulation)

    # protectedSplitEdges: optional set of frozenset({p1, p2}) coordinate-array
    # indices (as found in triangulation.triangles) that must never be split,
    # independent of blFlags -- unlike blFlags (which protects every edge of
    # a flagged triangle), this protects only the named edges, letting the
    # rest of their owning triangles refine normally. Absent for triangulations
    # that don't set it, so existing callers are unaffected.
    protectedSplitEdges = getattr(triangulation, 'protectedSplitEdges', None)

    # collect edges bigger in metric than sqrt(2) with their lengths
    lengths = []
    bigEdges = []   # stores idxs of big edges in master edge list
    for i in range(len(edges)):
        uIdx = edges[i]
        aTriStartIdx = uIdx - uIdx%3

        # blflags: skip edges that belong to BL triangles
        if triangulation.blFlags and triangulation.blFlags[aTriStartIdx//3] == 0:
            continue

        p1 = triangulation.triangles[uIdx]
        p2 = triangulation.triangles[aTriStartIdx + (uIdx%3+1)%3]

        # protectedSplitEdges: skip edges explicitly marked un-splittable
        if protectedSplitEdges and frozenset((p1, p2)) in protectedSplitEdges:
            continue

        segStart = [coords[p1], coords[p1+1]]
        segEnd = [coords[p2], coords[p2+1]]

        metricP1 = [triangulation.metricMesh[int(p1*1.5)], triangulation.metricMesh[int(p1*1.5)+1], triangulation.metricMesh[int(p1*1.5)+2]]
        metricP2 = [triangulation.metricMesh[int(p2*1.5)], triangulation.metricMesh[int(p2*1.5)+1], triangulation.metricMesh[int(p2*1.5)+2]]

        l = lengthInMetricMetric(metricP1, metricP2, segStart, segEnd)

        # # if edge is not on boundary but both endpoints on boundary, split this edge
        # if adjacents[uIdx] != -1 and pointLevels[p1//2] == 1 and pointLevels[p2//2] == 1:
        #     lengths.append(l)
        #     bigEdges.append(i)

        if l > upper_limit:
            lengths.append(l)
            bigEdges.append(i)

    print(f"Bigger edges fraction: {len(bigEdges)/len(edges):.2f}")
    # find sorting list according to metric length
    # sorted list is min -> max value
    order = heapSort(lengths)

    nSplits = 0
    # for all edges starting from biggest to smallest
    for i in range(len(order)-1, 0, -1):
        uIdx = edges[bigEdges[order[i]]]
        aTriStartIdx = uIdx - uIdx%3
        bTriStartIdx = adjacents[uIdx]

        p1 = triangulation.triangles[uIdx]
        p2 = triangulation.triangles[aTriStartIdx + (uIdx%3+1)%3]

        # protectedSplitEdges: skip edges explicitly marked un-splittable
        if protectedSplitEdges and frozenset((p1, p2)) in protectedSplitEdges:
            continue

        segStart = [coords[p1], coords[p1+1]]
        segEnd = [coords[p2], coords[p2+1]]

        metricP1 = [triangulation.metricMesh[int(p1*1.5)], triangulation.metricMesh[int(p1*1.5)+1], triangulation.metricMesh[int(p1*1.5)+2]]
        metricP2 = [triangulation.metricMesh[int(p2*1.5)], triangulation.metricMesh[int(p2*1.5)+1], triangulation.metricMesh[int(p2*1.5)+2]]

        l = lengthInMetricMetric(metricP1, metricP2, segStart, segEnd)

        if l < upper_limit: continue
        # if l < 2**0.5: continue

        # find new node
        newPoint, newMetric = splitEdge(triangulation, edges[bigEdges[order[i]]])

        # check triangle quality
        triQuality = splitTriQuality(triangulation, edges[bigEdges[order[i]]], newPoint, newMetric)

        # if triangle quality less, continue to next edge
        if triQuality < 1e-5: 
            # print('skip')
            continue
    
        # blflags: don't divide the edge which has BL triangles as neighbours
        if triangulation.blFlags and (
            triangulation.blFlags[aTriStartIdx//3] == 0 or
            (bTriStartIdx != -1 and triangulation.blFlags[bTriStartIdx//3] == 0)
        ):
            continue

        # split edge by subdividing on common edge
        # if internal edge
        if bTriStartIdx != -1:
            triangleSubdivide(triangulation, newPoint, [aTriStartIdx, bTriStartIdx], pointMetric=newMetric)
        # if boundary edge
        else:
            bdyTriangleSubdivide(triangulation, newPoint, uIdx, newMetric)
        nSplits += 1
    
    # swaps= localReconnection(triangulation, criterion="min-max-metric")
    print(f"Edges splitted: {nSplits}")

def collapseTriQuality(triangulation, uIdx, pc):
    coords = triangulation.coordinates
    triangles = triangulation.triangles
    metricMesh = triangulation.metricMesh

    # finding all triangles around pc
    ballOfPc = ballOfNode(triangulation, pc)

    # outer node
    po = triangulation.triangles[uIdx] 
    if pc == po:    # reversed edge
        aTriStartIdx = uIdx-uIdx%3
        po = triangles[aTriStartIdx+(uIdx+1)%3]

    qMin = 1.0

    # for all triangles around pc
    for i in range(len(ballOfPc)):
        triIdx = ballOfPc[i]

        # blFlags: skip quality check for BL triangles (and don't allow collapse that touches them)
        # if triangulation.blFlags and triangulation.blFlags[triIdx//3] == 0:
        #     return -1.0 

        triangleNodes = [triangles[triIdx], triangles[triIdx+1], triangles[triIdx+2]]

        # if triangle is collapsing, skip check
        if po in triangleNodes: continue

        # change pc in triangle to po for collapse
        pcIndex = triangleNodes.index(pc)
        triangleNodes[pcIndex] = po

        # check triangle quality
        # metric at all nodes
        m0 = [metricMesh[int(triangleNodes[0]*1.5)], metricMesh[int(triangleNodes[0]*1.5)+1], metricMesh[int(triangleNodes[0]*1.5)+2]]
        m1 = [metricMesh[int(triangleNodes[1]*1.5)], metricMesh[int(triangleNodes[1]*1.5)+1], metricMesh[int(triangleNodes[1]*1.5)+2]]
        m2 = [metricMesh[int(triangleNodes[2]*1.5)], metricMesh[int(triangleNodes[2]*1.5)+1], metricMesh[int(triangleNodes[2]*1.5)+2]]
        # determinants
        detM0 = det(m0)
        detM1 = det(m1)
        detM2 = det(m2)
        minDet = min(detM0, detM1, detM2)
        # area of triangle
        area = areab([coords[triangleNodes[0]], coords[triangleNodes[0]+1]],
                     [coords[triangleNodes[1]], coords[triangleNodes[1]+1]],
                     [coords[triangleNodes[2]], coords[triangleNodes[2]+1]])
        l01 = lengthInMetricMetric(m0, m1, [coords[triangleNodes[0]], coords[triangleNodes[0]+1]], [coords[triangleNodes[1]], coords[triangleNodes[1]+1]])
        l12 = lengthInMetricMetric(m1, m2, [coords[triangleNodes[1]], coords[triangleNodes[1]+1]], [coords[triangleNodes[2]], coords[triangleNodes[2]+1]])
        l20 = lengthInMetricMetric(m2, m0, [coords[triangleNodes[2]], coords[triangleNodes[2]+1]], [coords[triangleNodes[0]], coords[triangleNodes[0]+1]])
        q = 4/3**0.5 * 3 * area * minDet**0.5 / (l01*l01 + l12*l12 + l20*l20)

        qMin = min(qMin, q)

    return qMin

def collapseLenQuality(triangulation, uIdx, pc):
    coords = triangulation.coordinates
    triangles = triangulation.triangles

    # finding all nodes around pc
    nodes = nodesAroundNode(triangulation, pc)

    # outer node
    po = triangulation.triangles[uIdx] 
    if pc == po:    # reversed edge
        aTriStartIdx = uIdx-uIdx%3
        po = triangles[aTriStartIdx+(uIdx+1)%3]
    
    metricPo = [triangulation.metricMesh[int(po*1.5)], triangulation.metricMesh[int(po*1.5)+1], triangulation.metricMesh[int(po*1.5)+2]]
    segPo = [coords[po], coords[po+1]]

    minQuality = 1.0

    for i in range(len(nodes)):
        # skip collapsible edge
        if nodes[i] == po: continue
        pn = nodes[i]
        
        metricPn = [triangulation.metricMesh[int(pn*1.5)], triangulation.metricMesh[int(pn*1.5)+1], triangulation.metricMesh[int(pn*1.5)+2]]
        segPn = [coords[pn], coords[pn+1]]

        l = lengthInMetricMetric(metricPo, metricPn, segPo, segPn)

        if l < 1e-3 or l > 3:
            minQuality = -1.0
    
    return minQuality

# TODO: doesnt handle curved boundaries
def bdyEdgeCollapse(triangulation, pc, po):
    bdyPoints = triangulation.bdyPoints

    edgen = len(triangulation.meshObj['edgeSegments'])

    # finding collapse edge
    esIdx = None
    for i in range(edgen):
        edgeSeg = triangulation.meshObj['edgeSegments'][i]
        p1 = edgeSeg['p1']
        p2 = edgeSeg['p2']
        if p1 == pc//2 and p2 == po//2 or \
            p1 == po//2 and p2 == pc//2:
            esIdx = i
    collapseSeg = triangulation.meshObj['edgeSegments'][esIdx]
    p1c = collapseSeg['p1']
    p2c = collapseSeg['p2']
    
    # finding adjacent edge
    p1SegIdx = -1
    p2SegIdx = -1
    for i in range(edgen):
        edgeSeg = triangulation.meshObj['edgeSegments'][i]
        p1 = edgeSeg['p1']
        p2 = edgeSeg['p2']
        if p2 == p1c: 
            p1SegIdx = i
        if p1 == p2c:
            p2SegIdx = i
        if p1SegIdx != -1 and p2SegIdx != -1: break

    # changing adjacent edge segments
    adjP1Seg = triangulation.meshObj['edgeSegments'][p1SegIdx]
    adjP2Seg = triangulation.meshObj['edgeSegments'][p2SegIdx]

    if p1c == pc//2:
        adjP1Seg['p2'] = p2c
        adjP1Seg['sf2'] = collapseSeg['sf2']
        adjP1Seg['ednr2'] = collapseSeg['ednr2']
        adjP1Seg['dist2'] = collapseSeg['dist2']

    if p2c == pc//2:
        adjP2Seg['p1'] = p1c
        adjP2Seg['sf1'] = collapseSeg['sf1']
        adjP2Seg['ednr1'] = collapseSeg['ednr1']
        adjP2Seg['dist1'] = collapseSeg['dist1']

    # deleting collapsing edge
    triangulation.meshObj['edgeSegments'].pop(esIdx)

    bdyPoints.pop(esIdx*2)
    bdyPoints.pop(esIdx*2)


# collapsing from pc to po
def collapseEdge(triangulation, triangleIdxs, pc, po, pointLevels):
    coords = triangulation.coordinates
    triangles = triangulation.triangles
    adjacents = triangulation.adjacents
    metricMesh = triangulation.metricMesh
    metricLog = triangulation.metricLog

    aTri = triangleIdxs[0]
    bTri = triangleIdxs[1]
    
    # finding adjacents
    adjApc = None
    adjApo = None
    adjBpc = None
    adjBpo = None
    aNode = None
    bNode = None
    for i in range(3):
        au = triangles[aTri+i]
        av = triangles[aTri+(i+1)%3]

        bu = triangles[bTri+i]
        bv = triangles[bTri+(i+1)%3]

        if au == pc and av == po:
            adjApc = adjacents[aTri+(i+2)%3]
            adjApo = adjacents[aTri+(i+1)%3]
            aNode = triangles[aTri+(i+2)%3]
        elif au == po and av == pc:
            adjApc = adjacents[aTri+(i+1)%3]
            adjApo = adjacents[aTri+(i+2)%3]
            aNode = triangles[aTri+(i+2)%3]

        if bu == pc and bv == po:
            adjBpc = adjacents[bTri+(i+2)%3]
            adjBpo = adjacents[bTri+(i+1)%3]
            bNode = triangles[bTri+(i+2)%3]
        elif bu == po and bv == pc:
            adjBpc = adjacents[bTri+(i+1)%3]
            adjBpo = adjacents[bTri+(i+2)%3]
            bNode = triangles[bTri+(i+2)%3]

    # finding all triangles around pc and po which are affected
    ballOfPc = ballOfNode(triangulation, pc)
    ballOfPo = ballOfNode(triangulation, po)

    # cleaning coordinates of pc
    coords[pc] = None
    coords[pc+1] = None
    
    # cleaning point level
    pointLevels[pc//2] = None

    # cleaning metric at pc
    metricMesh[int(pc*1.5)] = -1
    metricMesh[int(pc*1.5)+1] = -1
    metricMesh[int(pc*1.5)+2] = -1
    metricLog[int(pc*1.5)] = -1
    metricLog[int(pc*1.5)+1] = -1
    metricLog[int(pc*1.5)+2] = -1
    
    # cleaning attached triangle at pc
    triangulation.adjPointTris[pc//2] = -1
    
    if aTri != -1:
        # cleaning triangle
        triangles[aTri] = -1
        triangles[aTri+1] = -1
        triangles[aTri+2] = -1

        # blFlags: update BL flags
        if triangulation.blFlags:
            triangulation.blFlags[aTri//3] = -1

        # cleaning adjacents
        adjacents[aTri] = None
        adjacents[aTri+1] = None
        adjacents[aTri+2] = None

        # changing adjacent of a-po adjacent triangle
        if adjApo != -1:
            for i in range(3):
                if adjacents[adjApo+i] == aTri:
                    # print(adjacents[adjAu+i], aTriStartIdx, adjVa)
                    adjacents[adjApo+i] = adjApc
                    break
        
        # changing attached triangle at a
        if triangulation.adjPointTris[aNode//2] == aTri:
            if adjApo != -1:
                triangulation.adjPointTris[aNode//2] = adjApo
            else:
                triangulation.adjPointTris[aNode//2] = adjApc
        
        if adjApo == -1 and adjApc == -1:
            print(ballOfPc, ballOfPo)
            raise RuntimeError
        # changing triangles around pc
        for i in range(len(ballOfPc)):
            if ballOfPc[i] == triangleIdxs[0] or ballOfPc[i] == triangleIdxs[1]: continue
            pcTri = ballOfPc[i]
            # for all nodes of triangle
            for j in range(3):
                if adjacents[pcTri+j] == aTri:
                    adjacents[pcTri+j] = adjApo
    
    if bTri != -1:
        # cleaning triangle
        triangles[bTri] = -1
        triangles[bTri+1] = -1
        triangles[bTri+2] = -1

        # blFlags: update BL flags
        if triangulation.blFlags:
            triangulation.blFlags[bTri//3] = -1

        # cleaning adjacents
        adjacents[bTri] = None
        adjacents[bTri+1] = None
        adjacents[bTri+2] = None

        # changing adjacent of b-po adjacent triangle
        if adjBpo != -1:
            for i in range(3):
                if adjacents[adjBpo+i] == bTri:
                    # print(adjacents[adjAu+i], aTriStartIdx, adjVa)
                    adjacents[adjBpo+i] = adjBpc
                    break
        
        # changing attached triangle at b
        if triangulation.adjPointTris[bNode//2] == bTri:
            if adjBpo != -1:
                triangulation.adjPointTris[bNode//2] = adjBpo
            else:
                triangulation.adjPointTris[bNode//2] = adjBpc

        # changing triangles around pc
        for i in range(len(ballOfPc)):
            if ballOfPc[i] == triangleIdxs[0] or ballOfPc[i] == triangleIdxs[1]: continue
            pcTri = ballOfPc[i]
    
            # for all nodes of triangle
            for j in range(3):
                if adjacents[pcTri+j] == bTri:
                    adjacents[pcTri+j] = adjBpo
    
    # changing triangles around pc
    for i in range(len(ballOfPc)):
        if ballOfPc[i] == triangleIdxs[0] or ballOfPc[i] == triangleIdxs[1]: continue
        pcTri = ballOfPc[i]
    
        # for all nodes of triangle
        for j in range(3):
            if triangles[pcTri+j] == pc:
                triangles[pcTri+j] = po
    
    # changing attached triangle at po
    if triangulation.adjPointTris[po//2] == aTri or triangulation.adjPointTris[po//2] == bTri:
        if aTri != -1 and adjApo != -1:
            triangulation.adjPointTris[po//2] = adjApo
        elif aTri != -1 and adjApc != -1:
            triangulation.adjPointTris[po//2] = adjApc  
        elif bTri != -1 and adjBpo != -1:
            triangulation.adjPointTris[po//2] = adjBpo
        elif bTri != -1 and adjBpc != -1:
            triangulation.adjPointTris[po//2] = adjBpc
        else:
            raise RuntimeError
     
    # changing boundary segments
    if aTri == -1 or bTri == -1:
        bdyEdgeCollapse(triangulation, pc, po)

    return True

def cleanTriangles(triangulation):
    coords = triangulation.coordinates
    triangles = triangulation.triangles
    adjacents = triangulation.adjacents
    metricMesh = triangulation.metricMesh
    metricLog = triangulation.metricLog
    adjPointTris = triangulation.adjPointTris
    bdyPoints = triangulation.bdyPoints

    # stores mapping of points idxs from old to new
    pointsMapping = [-1]*(len(coords)//2)

    # clean points
    newCoords = []
    newMetricMesh = []
    newMetricLog = []

    for i in range(0, len(coords), 2):
        # if valid point
        if coords[i] is not None and coords[i+1] is not None:
            # adding valid point
            newCoords.append(coords[i])
            newCoords.append(coords[i+1])
            # adding metric
            newMetricMesh.append(metricMesh[int(i*1.5)])
            newMetricMesh.append(metricMesh[int(i*1.5)+1])
            newMetricMesh.append(metricMesh[int(i*1.5)+2])
            newMetricLog.append(metricLog[int(i*1.5)])
            newMetricLog.append(metricLog[int(i*1.5)+1])
            newMetricLog.append(metricLog[int(i*1.5)+2])

            # mapping this point in new index
            pointsMapping[i//2] = len(newCoords) - 2

    # clean triangles
    newTriangles = []
    triMapping = [-1]*(len(triangles)//3)
    newBlFlags = []   # blFlags for new triangles, if blFlags exist in triangulation

    for i in range(0, len(triangles), 3):
        a, b, c = triangles[i], triangles[i+1], triangles[i+2]

        # if valid triangle
        if a != -1 and b != -1 and c != -1:
            newA = pointsMapping[a//2] 
            newB = pointsMapping[b//2]
            newC = pointsMapping[c//2]

            newTriangles.append(newA)
            newTriangles.append(newB)
            newTriangles.append(newC)

            triMapping[i//3] = len(newTriangles) - 3

            # blFlags: rebuild blFlags for new triangles
            if triangulation.blFlags:
                newBlFlags.append(triangulation.blFlags[i//3])

    # cleaning adjacents
    # newAdjacents = adjacencyList(newTriangles, len(newCoords)//2)
    
    bAdjacents = [-1]*len(newTriangles)
    for i in range(0, len(adjacents), 3):
        aAdj, bAdj, cAdj = adjacents[i], adjacents[i+1], adjacents[i+2]
        # if valid adjacent
        if aAdj is not None and bAdj is not None and cAdj is not None:
            # find where corrs triangle was mapped
            triIdx = triMapping[i//3]
            # fill mapped adjacents there
            bAdjacents[triIdx] = triMapping[adjacents[i]//3] if adjacents[i] != -1 else -1
            bAdjacents[triIdx+1] = triMapping[adjacents[i+1]//3] if adjacents[i+1] != -1 else -1
            bAdjacents[triIdx+2] = triMapping[adjacents[i+2]//3] if adjacents[i+2] != -1 else -1
    newAdjacents = bAdjacents  
    # if newAdjacents != bAdjacents:
    #     for i in range(len(newAdjacents)):
    #         print(newAdjacents[i], bAdjacents[i])
    #     raise ValueError

    # renumber boundary segments
    # loop trhough all edge segments in meshobj
    newBdyPoints = []
    edgen = len(triangulation.meshObj['edgeSegments'])
    for i in range(edgen):
        edgeSeg = triangulation.meshObj['edgeSegments'][i]
        p1 = edgeSeg['p1']
        p2 = edgeSeg['p2']
        p1New = pointsMapping[p1]//2
        p2New = pointsMapping[p2]//2
        triangulation.meshObj['edgeSegments'][i]['p1'] = p1New
        triangulation.meshObj['edgeSegments'][i]['p2'] = p2New

        newBdyPoints.append(p1New*2)
        newBdyPoints.append(p2New*2)
    
    triangulation.coordinates = newCoords
    triangulation.triangles = newTriangles
    triangulation.adjacents = newAdjacents
    triangulation.metricMesh = newMetricMesh
    triangulation.metricLog = newMetricLog
    triangulation.bdyPoints = newBdyPoints
    if triangulation.blFlags:   # blFlags: update blFlags for new triangles
        triangulation.blFlags = newBlFlags

    # build attached triangles
    triangulation.adjPointTris = triangulation.attachTriangles()

    # triangles have been renumbered, so the cached triangle index used as a
    # search starting point for metric() is no longer valid
    triangulation._prevTriIdx = 0

# blFlags: checks whether a node belongs to BL trinagle
def nodeTouchesBL(triangulation, node):
    triAroundNode = ballOfNode(triangulation, node)
    for tri in triAroundNode:
        if triangulation.blFlags and triangulation.blFlags[tri//3] == 0:
            return True
    return False

def recover_edges(edges: List[int], triangulation: 'Triangulation') -> List[Tuple[int, int]]:
    """
    Recovers edge node pairs from the edge indices produced by collectEdges.

    Parameters:
        edges (List[int]): List of global indices (uIdx) as returned by collectEdges.
        triangulation (Triangulation): The triangulation object containing triangles.

    Returns:
        List[Tuple[int, int]]: A list of (n1, n2) pairs where n1 < n2, representing each edge.
    """
    edge_pairs = []

    for uIdx in edges:
        # uIdx = i + j, where i is the triangle base index and j in {0, 1, 2}
        i = (uIdx // 3) * 3   # base index of the triangle
        j = uIdx % 3          # local edge index within the triangle

        u = int(triangulation.triangles[i + j])
        v = int(triangulation.triangles[i + (j + 1) % 3])

        edge_pairs.append((min(u, v), max(u, v)))

    return edge_pairs

def collapsePass(triangulation):
    coords = triangulation.coordinates
    triangles = triangulation.triangles
    adjacents = triangulation.adjacents
    edges = collectEdges(triangulation) # master list of edges
    pointLevels = classifyPoints(triangulation)  # 0 - internal, 1 - boundary, 2 - corner

    lengths = []
    shortEdges = []     # stores idxs of short edges in master edge list
    removalNodesFlags = [0]*(len(coords)//2)    # stores flags for each node which is connected with short edge and is considered for removal
    minNodeLength = [2]*(len(coords)//2)        # stores minimum edge length of all adjacent edges around that node   

    for i in range(len(edges)):
        uIdx = edges[i]
        aTriStartIdx = uIdx - uIdx%3
        
        p1 = triangulation.triangles[uIdx]
        p2 = triangulation.triangles[aTriStartIdx + (uIdx%3+1)%3]

        segStart = [coords[p1], coords[p1+1]]
        segEnd = [coords[p2], coords[p2+1]]

        if triangulation.metricMesh:
            metricP1 = [triangulation.metricMesh[int(p1*1.5)], triangulation.metricMesh[int(p1*1.5)+1], triangulation.metricMesh[int(p1*1.5)+2]]
            metricP2 = [triangulation.metricMesh[int(p2*1.5)], triangulation.metricMesh[int(p2*1.5)+1], triangulation.metricMesh[int(p2*1.5)+2]]
            l = lengthInMetricMetric(metricP1, metricP2, segStart, segEnd)
            if l < lower_limit:
                lengths.append(l)
                shortEdges.append(i)
                removalNodesFlags[p1//2] = 1    # both p1 and p2 points become collapse candidates
                minNodeLength[p1//2] = min(minNodeLength[p1//2], l) # stores minimum edge length around this node for later sorting of collapse candidates
                removalNodesFlags[p2//2] = 1
                minNodeLength[p2//2] = min(minNodeLength[p2//2], l)                    
        else:
            l = lengthInMetric(triangulation.metric, segStart, segEnd)
            if l < lower_limit:
                lengths.append(l)
                shortEdges.append(i)
                removalNodesFlags[p1//2] = 1
                minNodeLength[p1//2] = min(minNodeLength[p1//2], l)
                removalNodesFlags[p2//2] = 1
                minNodeLength[p2//2] = min(minNodeLength[p2//2], l)
    print(f"Shorter edges fraction: {len(shortEdges)/len(edges):.2f}")

    # collect removal nodes
    removalNodes = []           # stores idxs of removal nodes in coords
    removalNodeLengths = []     # stores minimum edge length around that node
    for i in range(len(removalNodesFlags)):
        if removalNodesFlags[i] == 1:
            removalNodes.append(i*2)
            removalNodeLengths.append(minNodeLength[i])

    # sort removal nodes by their minimum adjacent edge length
    order = heapSort(removalNodeLengths)  # stores idxs of removal nodes in sorted order of their minimum adjacent edge length

    # for all removal nodes
    nCollapses = 0
    for i in range(len(removalNodes)):
        pc = removalNodes[order[i]] # candidate centered node
        if removalNodesFlags[pc//2] == 0: continue  # check if node is active (can be deactivated by earlier collapse)
        if nodeTouchesBL(triangulation, pc): continue   # blFlags
        
        # collect all edges around candiate node
        # Warning: here uIdx only represents edge index in triangles. It doesnt mean how this edge is oriented wrt pc and po.
        uIdxs = edgesAroundNode(triangulation, pc)

        # find metric length of each edge
        adjEdgeLengths = []
        for j in range(len(uIdxs)):
            uIdx = uIdxs[j]
            aTriStartIdx = uIdx - uIdx%3
            p1 = triangulation.triangles[uIdx]
            p2 = triangulation.triangles[aTriStartIdx + (uIdx%3+1)%3]

            segStart = [coords[p1], coords[p1+1]]
            segEnd = [coords[p2], coords[p2+1]]

            metricP1 = [triangulation.metricMesh[int(p1*1.5)], triangulation.metricMesh[int(p1*1.5)+1], triangulation.metricMesh[int(p1*1.5)+2]]
            metricP2 = [triangulation.metricMesh[int(p2*1.5)], triangulation.metricMesh[int(p2*1.5)+1], triangulation.metricMesh[int(p2*1.5)+2]]

            l = lengthInMetricMetric(metricP1, metricP2, segStart, segEnd)
            adjEdgeLengths.append(l)

        orderAdjEdge = heapSort(adjEdgeLengths) # sort adjacent edges

        # for all edges around pc starting from smallest to biggest
        for j in range(len(uIdxs)):
            if removalNodesFlags[pc//2] == 0: continue  # check if node is active (can be deactivated by earlier collapse)
            uIdx = uIdxs[orderAdjEdge[j]]
            po = triangulation.triangles[uIdx]  # outer node
            
            if pc == po:    # reversed edge
                aTriStartIdx = uIdx-uIdx%3
                po = triangulation.triangles[aTriStartIdx+(uIdx+1)%3]

            if nodeTouchesBL(triangulation, po): continue   # blFlags
            
            # if boundary point collapsing to internal point, skip
            if pointLevels[pc//2] == 1 and pointLevels[po//2] == 0: continue

            # if boundary segment end point (singular) collapsing to any other point, skip
            if pointLevels[pc//2] == 2: continue

            # if both endpoints of edge are on boundary and are on different geometric boundaries, cannot collapse
            # TODO: eliminate this
            if pointLevels[pc//2] == 1 and pointLevels[po//2] == 1:
                # check boundary segment number
                pcSeg = findEdgeSegment(triangulation, pc)
                poSeg = findEdgeSegment(triangulation, po)
                if triangulation.meshObj['edgeSegments'][pcSeg]['ednr1'] != \
                triangulation.meshObj['edgeSegments'][poSeg]['ednr1']: continue

            triQuality = collapseTriQuality(triangulation, uIdx, pc)    # check collapse quality
            lenQuality = collapseLenQuality(triangulation, uIdx, pc)    # check collapse length quality
            if triQuality < 1e-3 or lenQuality < 0: continue
            
            aTri = uIdx-uIdx%3
            bTri = adjacents[uIdx]

            # a boundary point (pc) can only be safely removed by collapsing it
            # along one of its own two boundary edges (aTri == -1 or bTri == -1):
            # only then does bdyEdgeCollapse run below to reattach pc's
            # meshObj['edgeSegments']/bdyPoints entries to po. Collapsing pc away
            # via an interior edge leaves pc's real boundary edges pointing at a
            # removed node -- cleanTriangles then silently renumbers that stale
            # reference to -1 (pointsMapping[pc] // 2, since -1 // 2 == -1 in
            # Python), corrupting edgeSegments and causing a later findEdgeSegment
            # RuntimeError in splitPass/bdyTriangleSubdivide.
            if pointLevels[pc//2] != 0 and aTri != -1 and bTri != -1: continue

            # blFLags: BL triangle protection
            if triangulation.blFlags:
                if triangulation.blFlags[aTri//3] == 0: continue
                if bTri != -1 and triangulation.blFlags[bTri//3] == 0: continue

            p1 = triangulation.triangles[uIdx]
            p2 = triangulation.triangles[aTri + (uIdx%3+1)%3]

            segStart = [coords[p1], coords[p1+1]]
            segEnd = [coords[p2], coords[p2+1]]

            metricP1 = [triangulation.metricMesh[int(p1*1.5)], triangulation.metricMesh[int(p1*1.5)+1], triangulation.metricMesh[int(p1*1.5)+2]]
            metricP2 = [triangulation.metricMesh[int(p2*1.5)], triangulation.metricMesh[int(p2*1.5)+1], triangulation.metricMesh[int(p2*1.5)+2]]

            l = lengthInMetricMetric(metricP1, metricP2, segStart, segEnd)

            if l > lower_limit: continue

            # blFlags
            if triangulation.blFlags:
                if aTri != -1 and triangulation.blFlags[aTri//3] == 0:
                    print("WARNING: collapsing BL triangle", aTri)
                if bTri != -1 and triangulation.blFlags[bTri//3] == 0:
                    print("WARNING: collapsing BL triangle", bTri)

            success = collapseEdge(triangulation, [aTri, bTri], pc, po, pointLevels)
            
            if success:
                removalNodesFlags[po//2] = 0
                removalNodesFlags[pc//2] = 0
                adjNodes = nodesAroundNode(triangulation, po)
                for node in adjNodes:   # make all nodes adjacent to po inactive
                    removalNodesFlags[node//2] = 0
                nCollapses += 1
                break   # if collapse is successful, break and move to next removal node
            else:
                continue    # if collapse failed, try next adjacent edge
    print(f"Edges collapsed: {nCollapses}")


def edgePrimitiveAdapt(triangulation, iterations):
    from .pointQuadTree import buildPointQuadTree
    from .aflr import localReconnection, checkFlatTriangles

    for i in range(iterations):
        print(f"pass {i}")
        splitPass(triangulation)
        collapsePass(triangulation)
        cleanTriangles(triangulation)   # cleaning collapsed triangles and renumbering points and triangles
        triangulation.quadTree = buildPointQuadTree(triangulation)
        triangulation.elementADT = buildElementADT(triangulation)
        swaps = localReconnection(triangulation, criterion="min-max-metric", skipTriangleFlags=triangulation.blFlags)
        print(f"Swaps: {swaps}")
    checkFlatTriangles(triangulation.coordinates, triangulation.triangles)
    return triangulation
