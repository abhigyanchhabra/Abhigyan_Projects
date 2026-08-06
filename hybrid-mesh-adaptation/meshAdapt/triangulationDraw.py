import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.patches import Polygon, Rectangle, Ellipse
from matplotlib.backend_bases import MouseEvent
from typing import TYPE_CHECKING, List, Callable, Optional, Dict
if TYPE_CHECKING:
    from .triangulation import Triangulation

from .metric import eigen, normOfVec, lengthInMetricMetric, lengthInMetric, lengthInMetricGaussian
from .triangulationSearch import searchTriangle, collectEdges

# draw functions
def drawTriangulation(triangulation: 'Triangulation', axis: Optional[plt.Axes]=None, flags: Optional[List[int]]=None) -> plt.Axes:
    """
    Draws a triangulation on a given Matplotlib axis.

    Parameters:
        triangulation (Triangulation): The triangulation object containing the list of triangles and coordinates.
        axis (Optional[plt.Axes]): The Matplotlib axis where the triangulation will be drawn. If None, the current axis will be used.
        flags (Optional[List[int]]): A list of flags indicating whether a triangle should be filled with color. Defaults to None.

    Returns:
        plt.Axes: The axis object with the triangulation plot.
    """
    triangles = triangulation.triangles
    coords = triangulation.coordinates

    if axis is None:
        axis = plt.gca()

    # for all triangles
    for triangleIdx in range(0, len(triangles), 3):
        # collect triangle vertices
        triangleVertices = [(coords[triangles[triangleIdx]], coords[triangles[triangleIdx]+1]),
                            (coords[triangles[triangleIdx+1]], coords[triangles[triangleIdx+1]+1]),
                            (coords[triangles[triangleIdx+2]], coords[triangles[triangleIdx+2]+1])]
        
        # fill accordingly if flags are available
        fill = 'none'
        if flags:
            if flags[int(triangleIdx/3)] == 1:
                fill = 'lightgray'
        triangle = Polygon(triangleVertices, closed=True, facecolor=fill, edgecolor='black', joinstyle='bevel')
        
        # add triangle to plot axis
        axis.add_patch(triangle)

    axis.figure.canvas.mpl_connect('button_press_event', partial(onTriangulationClick, triangulation))
    axis.set_aspect('equal')
    axis.margins(0.02)
    
    return axis

def drawTriangle(triangulation: 'Triangulation', triangleIndex: int, axis: Optional[plt.Axes]=None) -> plt.Axes:
    """
    Draws a single triangle from a triangulation on a given Matplotlib axis.

    Parameters:
        triangulation (Triangulation): The triangulation object containing the list of triangles and coordinates.
        triangleIndex (int): The index of the triangle in triangles list to draw in the triangulation.
        axis (Optional[plt.Axes], optional): The Matplotlib axis where the triangle will be drawn. If None, the current axis will be used.

    Returns:
        plt.Axes: The axis object with the drawn triangle.
    """
    t = triangleIndex*3
    if t >= len(triangulation.triangles): raise IndexError('Triangle index out of bound.')
    triangles = triangulation.triangles
    coords = triangulation.coordinates

    if axis is None:
        axis = plt.gca()

    # draw triangulation
    # drawTriangulation(triangulation, axis=axis)

    # for given triangle
    triangleVertices = [(coords[triangles[t]], coords[triangles[t]+1]),
                        (coords[triangles[t+1]], coords[triangles[t+1]+1]),
                        (coords[triangles[t+2]], coords[triangles[t+2]+1])]
    triangle = Polygon(triangleVertices, closed=True, facecolor='lightgray', edgecolor='red', alpha=0.5)

    # add triangle to plot axis
    axis.add_patch(triangle)

    # zoom on given triangle
    # _focusTriangle(triangulation, triangleIndex, axis)

    plt.draw()
    
    return axis

# specifically used for edge already present in triangulation
def drawSegment(triangulation, triangleIndex, edgeIndex, axis=None):
    t = triangleIndex*3
    triangles = triangulation.triangles
    coords = triangulation.coordinates

    segStartIndex = triangles[t + edgeIndex]
    segEndIndex = triangles[t + (edgeIndex+1)%3]

    if axis is None:
        axis = plt.gca()

    # draw triangulation
    drawTriangulation(triangulation, axis=axis)
    axis.plot([coords[segStartIndex], coords[segEndIndex]], [coords[segStartIndex+1], coords[segEndIndex+1]], color='lime')

    # zoom on given triangle
    _focusTriangle(triangulation, triangleIndex, axis)

    return axis

def drawMetricBalls(
        triangulation: 'Triangulation', 
        metricFunction: Optional[Callable[[float, float], List[float]]]=None, 
        axis: Optional[plt.Axes]=None, 
        drawMesh: Optional[bool]=False, 
        scale: Optional[float]=1) -> plt.Axes:
    """
    Draws unit circles (metric balls) in a given metric field on a triangulation.

    Parameters:
        triangulation (Triangulation): The triangulation object containing the list of coordinates and metric information.
        metricFunction (Optional[Callable[[float, float], List[float]]]): A function that computes the metric tensor at a given point. 
            If None, the `metricMesh` from the triangulation will be used.
        axis (Optional[plt.Axes]): The Matplotlib axis where the metric balls will be drawn. If None, the current axis will be used.
        drawMesh (Optional[bool]): A flag indicating whether the triangulation mesh should be drawn as a background. Defaults to False.
        scale (Optional[float]): A scaling factor for the size of the metric balls. Defaults to 1.

    Returns:
        plt.Axes: The axis object with the drawn metric balls.
    """
    coords = triangulation.coordinates

    if axis is None:
        axis = plt.gca()
    
    if drawMesh:
        drawTriangulation(triangulation, axis=axis)
    
    # for all points
    for pointIdx in range(0, len(coords), 2):
        # fetching metric at this point
        if metricFunction == None:
            metric = [triangulation.metricMesh[pointIdx*3//2], triangulation.metricMesh[pointIdx*3//2+1], triangulation.metricMesh[pointIdx*3//2+2]]
        else:
            metric = metricFunction(coords[pointIdx], coords[pointIdx+1])
            
        # eigenvalues
        eigenObject = eigen(metric)
        lambda1 = eigenObject['lambda1']
        lambda2 = eigenObject['lambda2']

        width = 1/lambda1**0.5
        height = 1/lambda2**0.5

        # angle
        v1 = eigenObject['vec1']
        v2 = eigenObject['vec2']
        angle = np.degrees(np.arctan2(v1[1], v1[0]))

        # draw ellipses
        ellipse = Ellipse([coords[pointIdx], coords[pointIdx+1]], width*scale*2, height*scale*2, angle=angle, edgecolor='grey', facecolor='none', linewidth=0.5)
        axis.add_patch(ellipse)

    # focus
    axis.set_xlim([np.min(coords[::2]), np.max(coords[::2])])
    axis.set_ylim([np.min(coords[1::2]), np.max(coords[1::2])])
    axis.set_aspect('equal')
    axis.margins(0.02)
    
    return axis

def impliedMetricBalls(
        imetricPath: str, 
        triangulation: 'Triangulation', 
        ballScale: Optional[float]=1, 
        skipTriangles: Optional[int]=0, 
        drawMesh: Optional[bool]=False,  
        axis: Optional[plt.Axes]=None) -> plt.Axes:
    """
    Draws unit circles (metric balls) at the centroids of triangles in a triangulation, based on an implied metric.

    Parameters:
        imetricPath (str): The path to the file containing the implied metric data.
        triangulation (Triangulation): The triangulation object containing the list of coordinates and triangles.
        ballScale (float, optional): A scaling factor for the size of the metric balls. Defaults to 1.
        skipTriangles (int, optional): An integer for skipping triangles when drawing (defaults to 0, which means no skipping).
        drawMesh (bool, optional): A flag indicating whether the triangulation mesh should be drawn as a background. Defaults to False.
        axis (Optional[plt.Axes], optional): The Matplotlib axis where the metric balls will be drawn. If None, the current axis will be used.

    Returns:
        plt.Axes: The axis object with the drawn metric balls.
    """
    # read implied metric file
    imetric = []
    with open(imetricPath, 'r') as file:
        for i, line in enumerate(file):
            # skipping header line
            if i > 0:
                a, b, c = line.strip().split()
                imetric.append(float(a))
                imetric.append(float(b))
                imetric.append(float(c))

    if axis is None:
        axis = plt.gca()
    
    if drawMesh:
        drawTriangulation(triangulation, axis=axis)

    coords = triangulation.coordinates
    triangles = triangulation.triangles

    # for all triangles
    for i in range(0, len(triangles), 3):
        ux = coords[triangles[i]]
        uy = coords[triangles[i]+1]
        vx = coords[triangles[i+1]]
        vy = coords[triangles[i+1]+1]
        wx = coords[triangles[i+2]]
        wy = coords[triangles[i+2]+1]

        # finding centroid of triangle
        cx = 1/3 * (ux+vx+wx)
        cy = 1/3 * (uy+vy+wy)

        # fetching imetric at this point
        metric = [imetric[i], imetric[i+1], imetric[i+2]]

        # eigenvalues
        eigenObject = eigen(metric)
        lambda1 = eigenObject['lambda1']
        lambda2 = eigenObject['lambda2']

        width = 1/lambda1**0.5 * 2
        height = 1/lambda2**0.5 * 2

        # angle
        v1 = eigenObject['vec1']
        v2 = eigenObject['vec2']
        angle = np.degrees(np.arctan2(v1[1], v1[0]))

        # draw ellipses
        ellipse = Ellipse([cx, cy], width*ballScale, height*ballScale, angle=angle, edgecolor='r', facecolor='none', linewidth=0.5)
        skipTriangles = 1/3 if skipTriangles == 0 else skipTriangles
        if i%(skipTriangles*3) == 0:
            axis.add_patch(ellipse)

    # focus
    axis.set_xlim([np.min(coords[::2]), np.max(coords[::2])])
    axis.set_ylim([np.min(coords[1::2]), np.max(coords[1::2])])
    axis.set_aspect('equal')
    axis.margins(0.02)

    return axis

# Mesh Quality Functions

def angleHistogram(
        triangulation: 'Triangulation', 
        label: Optional[str]=None, 
        maxAngle: Optional[bool]=False, 
        axis: Optional[plt.Axes]=None, 
        width: Optional[float]=0.3, 
        offset: Optional[float]=0, 
        binLabel: Optional[bool]=False) -> plt.Axes:
    """
    Generates a histogram of the angles of the edges in the given triangulation.

    This function computes the angles of all edges in the triangulation and creates a histogram of these angles. 
    If the `maxAngle` flag is set to True, the largest angle in each triangle is recorded. The histogram is plotted 
    on the provided axis, or on the current axis if none is provided. The function also supports optional labeling 
    of the histogram bins.

    Parameters:
        triangulation (Triangulation): The triangulation object containing the list of triangles and coordinates.
        label (Optional[str], optional): The label for the histogram in the plot legend. Defaults to None.
        maxAngle (bool, optional): A flag indicating whether to only record the maximum angle in each triangle. Defaults to False.
        axis (Optional[plt.Axes], optional): The Matplotlib axis on which the histogram will be plotted. If None, the current axis will be used.
        width (float, optional): The width of the bars in the histogram. Defaults to 0.3.
        offset (float, optional): The offset for positioning the bins along the x-axis. Defaults to 0.
        binLabel (bool, optional): A flag indicating whether to label the bins. Defaults to False.

    Returns:
        plt.Axes: The axis object with the drawn histogram.
    """
    coords = triangulation.coordinates
    angles = []
    # for all triangles
    lenTriangles = len(triangulation.triangles)
    for i in range(0, lenTriangles, 3):
        maxTheta = 0
        # for all 3 triangle edges
        for j in range(3):
            a = triangulation.triangles[i+j]
            b = triangulation.triangles[i+(j+1)%3]
            c = triangulation.triangles[i+(j+2)%3]

            coordsA = [coords[a], coords[a+1]]
            coordsB = [coords[b], coords[b+1]]
            coordsC = [coords[c], coords[c+1]]

            ba = [coordsA[0]-coordsB[0], coordsA[1]-coordsB[1]]
            bc = [coordsC[0]-coordsB[0], coordsC[1]-coordsB[1]]

            dot = ba[0]*bc[0] + ba[1]*bc[1]

            normBa = normOfVec(ba)
            normBc = normOfVec(bc)

            theta = np.arccos(np.clip(dot/(normBa*normBc), -1.0, 1.0))*180/3.141592
            if maxAngle: 
                if theta > maxTheta: maxTheta = theta
            else:
                angles.append(theta)
        
        if maxAngle: angles.append(maxTheta)
    
    # histogram for edge lengths
    anglesNp = np.array(angles)
    binAngles = np.linspace(-7.5, 187.5, 14, endpoint=True)
    counts, binEdges = np.histogram(anglesNp, bins=binAngles)
    counts = counts/len(angles)
    binCenters = (binAngles[:-1] + binAngles[1:])/2
    binLabel = binCenters.copy()

    # offset histogram
    binCenters += (binCenters[1]-binCenters[0])*offset
    if axis is None:
        axis = plt.gca()

    # Create labels for bin ranges like '0-10', '10-20', etc.
    # binLabels = [f'{int(binEdges[i])}-{int(binEdges[i+1])}' for i in range(len(binEdges)-1)]

    # Set the x-ticks to be the bin midpoints and label them as the bin ranges
    axis.set_xticks(binLabel)  # Set ticks at the middle of each bin
    axis.set_xticklabels(binLabel, rotation=45)  # Use the bin ranges as labels

    axis.bar(binCenters, counts, width=(binCenters[1]-binCenters[0])*width, edgecolor='black', label=label)

    plt.legend()
    return axis

def edgeLengthsHistogram(
        triangulation: 'Triangulation', 
        quadrature: Optional[bool]=False, 
        axis: Optional[plt.Axes]=None) -> plt.Axes:
    """
    Generates a histogram of the lengths of edges in the given triangulation.

    Parameters:
        triangulation (Triangulation): The triangulation object containing the list of edges and coordinates.
        quadrature (Optional[bool], optional): A flag indicating whether to use Gaussian quadrature for length calculation. Defaults to None.
        axis (Optional[plt.Axes], optional): The Matplotlib axis where the histogram will be plotted. If None, the current axis will be used.

    Returns:
        plt.Axes: The axis object with the drawn histogram.
    """
    coords = triangulation.coordinates
    # edges = triangulation.edges
    edges = collectEdges(triangulation)
    lengths = []

    # for all edges, find metric length
    lengths = []
    for i in range(0, len(edges), 2):
        uIdx = edges[i]
        aTriStartIdx = uIdx - uIdx%3
        p1 = triangulation.triangles[uIdx]
        p2 = triangulation.triangles[aTriStartIdx + (uIdx%3+1)%3]
        
        segStart = [coords[p1], coords[p1+1]]
        segEnd = [coords[p2], coords[p2+1]]

        if quadrature:
            l = lengthInMetricGaussian(triangulation.metric, segStart, segEnd)
        else:
            if triangulation.metricMesh:
                metricP1 = [triangulation.metricMesh[int(p1*1.5)], triangulation.metricMesh[int(p1*1.5)+1], triangulation.metricMesh[int(p1*1.5)+2]]
                metricP2 = [triangulation.metricMesh[int(p2*1.5)], triangulation.metricMesh[int(p2*1.5)+1], triangulation.metricMesh[int(p2*1.5)+2]]

                l = lengthInMetricMetric(metricP1, metricP2, segStart, segEnd)
            else:
                l = lengthInMetric(triangulation.metric, segStart, segEnd)
        
        lengths.append(l)

    # histogram for edge lengths
    lengthsNp = np.array(lengths)
    binEdges = np.linspace(0, 2, 41)
    counts, _ = np.histogram(lengthsNp, bins=binEdges)
    binCenters = (binEdges[:-1] + binEdges[1:])/2

    if axis is None:
        axis = plt.gca()

    axis.bar(binCenters, counts, width=binCenters[1]-binCenters[0], edgecolor='black')
    
    return axis

def onTriangulationClick(triangulation: 'Triangulation', event: MouseEvent) -> None:
    """
    Handles the click event for interacting with the triangulation plot.

    This function is triggered when clicked on the triangulation plot. If the click is within the bounds of 
    the plot and is a right-click, it finds the triangle that contains the clicked point, draws the triangle, and 
    displays various information about the triangle element, including its points, adjacent triangles, and metric information 
    (if available). The metric lengths of the edges are also computed using both geometric and Gaussian quadrature methods.

    Parameters:
        triangulation (Triangulation): The triangulation object containing the list of triangles, coordinates, 
                                        and adjacent triangle information.
        event (plt.MouseEvent): The event object containing information about the mouse click (e.g., position, 
                                button clicked).

    Returns:
        None: This function doesn't return anything. It updates the plot and prints information to the console.
    """
    # Check if the click was inside the axis
    if event.inaxes:
        # if right click
        if event.button == 3:
            # Get the x and y coordinates in data space
            x, y = event.xdata, event.ydata
            coords = triangulation.coordinates
            triangles = triangulation.triangles
            adjacents = triangulation.adjacents

            triangleIdx = searchTriangle(x, y, triangulation)
            if triangleIdx == -1: 
                print("---------------Point outside mesh---------------")
            else:
                if type(triangleIdx) == list: triangleIdx = triangleIdx[0]
                drawTriangle(triangulation, triangleIdx//3)
                print(f"Element {triangleIdx} Information:")
                print("---------------Points---------------")
                point0 = [coords[triangles[triangleIdx]], coords[triangles[triangleIdx]+1]]
                point1 = [coords[triangles[triangleIdx+1]], coords[triangles[triangleIdx+1]+1]]
                point2 = [coords[triangles[triangleIdx+2]], coords[triangles[triangleIdx+2]+1]]
                print("Point 0 coordinates: {0}".format(point0))
                print("Point 1 coordinates: {0}".format(point1))
                print("Point 2 coordinates: {0}".format(point2))
                print("---------------Adjacents---------------")
                print("Adjacent 1: {0}, Adjacent 1: {1}, Adjacent 1: {2}".format(adjacents[triangleIdx], adjacents[triangleIdx+1], adjacents[triangleIdx+2]))
                if triangulation.metricMesh:
                    metricMesh = triangulation.metricMesh
                    metric0 = triangulation.metric(point0[0], point0[1])
                    metric1 = triangulation.metric(point1[0], point1[1])
                    metric2 = triangulation.metric(point2[0], point2[1])
                    print("---------------Metric---------------")
                    print("Metric at 0: {0}".format(metric0))
                    print("Metric at 1: {0}".format(metric1))
                    print("Metric at 2: {0}".format(metric2))
                    print("---------------Metric Lengths (Geometric Approx.)---------------")
                    print(f"01 metric length: {lengthInMetricMetric(metric0, metric1, point0, point1)}")
                    print(f"12 metric length: {lengthInMetricMetric(metric1, metric2, point1, point2)}")
                    print(f"20 metric length: {lengthInMetricMetric(metric2, metric0, point2, point0)}")
                if triangulation.bgMetricFunction:
                    print("---------------Metric Lengths (Gaussian Quad.)---------------")
                    print(f"01 metric length: {lengthInMetricGaussian(triangulation.bgMetricFunction, point0, point1)}")
                    print(f"12 metric length: {lengthInMetricGaussian(triangulation.bgMetricFunction, point1, point2)}")
                    print(f"20 metric length: {lengthInMetricGaussian(triangulation.bgMetricFunction, point2, point0)}")

                # quadTree
                # triIdx = qt.searchPoint(x, y)
                # drawTriangle(tri, triIdx//3)
                # qt = triangulation.quadTree
                # leafNode = qt._findLeafNode(x, y)
                # if leafNode:
                # #     # print(leafNode.triangles)
                #     bounds = leafNode.bounds
                #     rect = Rectangle((bounds[0], bounds[1]), abs(bounds[0]-bounds[2]), abs(bounds[1]-bounds[3]), linewidth=1, edgecolor='r', facecolor='r', alpha=0.3)
                #     plt.gca().add_patch(rect)
                #     for point in leafNode.points:
                #         px = qt.globalCoordinates[point]
                #         py = qt.globalCoordinates[point+1]
                #         print(px, py)
                #         plt.gca().scatter(px, py)
                #     # px, py = qt.searchNearest(x, y)
                #     # ax.scatter(px, py)
                #     plt.draw()

# internal use only
def _focusTriangle(triangulation, triangleIndex, axis):
    t = triangleIndex*3
    triangles = triangulation.triangles
    coords = triangulation.coordinates

    triangleVertices = [(coords[triangles[t]], coords[triangles[t]+1]),
                        (coords[triangles[t+1]], coords[triangles[t+1]+1]),
                        (coords[triangles[t+2]], coords[triangles[t+2]+1])]

    # variables for setting zoom
    triangleX = [x[0] for x in triangleVertices]
    triangleY = [x[1] for x in triangleVertices]
    triangleDx = max(triangleX) - min(triangleX)
    triangleDy = max(triangleY) - min(triangleY)

    axis.set_aspect('equal')
    if triangleDx > triangleDy:
        axis.set_xlim([min(triangleX), max(triangleX)])
        axis.set_ylim([min(triangleY) + triangleDy*0.5 - triangleDx*0.5, min(triangleY) + triangleDy*0.5 + 0.5*triangleDx])
    else:
        axis.set_ylim([min(triangleY), max(triangleY)])
        axis.set_xlim([min(triangleX) + triangleDx*0.5 - triangleDy*0.5, min(triangleX) + triangleDx*0.5 + 0.5*triangleDy])

def drawAspectRatioField(triangulation, grad=False, scalarMin=None, scalarMax=None, axis=None):
    from matplotlib.tri import Triangulation as Tri
    from matplotlib.tri import LinearTriInterpolator

    def compute_gradient_magnitude(triang, z):
        """
        Compute the gradient magnitude of scalar field z on triangulation triang.
        Returns the gradient magnitude at each vertex of the triangulation.
        """
        # Create a linear interpolator
        interp = LinearTriInterpolator(triang, z)
        
        # Get the gradients at each vertex
        dx, dy = interp.gradient(triang.x, triang.y)
        
        # Compute the gradient magnitude
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        
        # Handle any NaN values (which can occur at the boundary)
        gradient_magnitude = np.nan_to_num(gradient_magnitude)
        
        return gradient_magnitude

    xs = np.array(triangulation.coordinates[::2])
    ys = np.array(triangulation.coordinates[1::2])

    # Scalar values at each vertex
    scalar_values = []
    for i in range(len(xs)):
        metric = triangulation.metric(xs[i], ys[i])
        eigenObj = eigen(metric)
        ar = max(eigenObj['lambda1'], eigenObj['lambda2'])/min(eigenObj['lambda1'], eigenObj['lambda2'])
        scalar_values.append(ar)
    scalar_values = np.array(scalar_values)

    triangles = []
    for i in range(0, len(triangulation.triangles), 3):
        triangles.append([triangulation.triangles[i]//2, triangulation.triangles[i+1]//2, triangulation.triangles[i+2]//2])
    triangles = np.array(triangles)

    # Create a triangulation object
    triang = Tri(xs, ys, triangles)

    if grad:
        gradMag = compute_gradient_magnitude(triang, scalar_values)

    if axis is None:
        axis = plt.gca()

    # Plot the triangulation with colors based on scalar values
    # Use tripcolor for filling triangles with interpolated colors
    if grad:
        tcf = axis.tripcolor(triang, gradMag, shading='gouraud', cmap='viridis', vmin=scalarMin, vmax=scalarMax)
    else:
        tcf = axis.tripcolor(triang, scalar_values, shading='gouraud', cmap='viridis', vmin=scalarMin, vmax=scalarMax)

    # Add a colorbar
    cbar = axis.figure.colorbar(tcf, ax=axis)
    if grad:
        cbar.set_label('Aspect Ratio Gradient')
    else:
        cbar.set_label('Aspect Ratio')

    # Set labels and title
    axis.set_aspect('equal')

    plt.tight_layout()

    return axis
