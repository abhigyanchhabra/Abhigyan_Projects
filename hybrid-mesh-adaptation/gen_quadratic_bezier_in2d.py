import os

import numpy as np
from scipy.interpolate import CubicSpline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of this script, used to anchor output_file's path

output_file = os.path.join(SCRIPT_DIR, "RAE2822.in2d")  # destination path for the generated .in2d mesh-geometry file
n_farfield = 80      # number of vertices (= line segments) of the farfield polygon
r_farfield = 100.0   # circumradius of the farfield polygon, centred at the origin
airfoil_file = os.path.join(SCRIPT_DIR, "airfoil_pts.txt")  # source file with raw airfoil surface coordinates


def read_lednicer_airfoil(filepath):
    """
    Parse a two-block airfoil coordinate file (Lednicer-style, e.g.
    airfoil_pts.txt): a title line, a counts line 'n_upper  n_lower',
    then the upper-surface points (LE -> TE) and the lower-surface
    points (LE -> TE), each block separated by a blank line.

    Returns
    -------
    upper, lower : ndarray, shape (n_upper, 2) / (n_lower, 2)
    """
    with open(filepath, encoding="utf-8") as f:
        rows = [line.split() for line in f if line.strip()]  # tokenized non-blank lines of the airfoil file

    n_upper, n_lower = (int(round(float(v))) for v in rows[1])  # point counts for each surface, read from the counts line
    coords = rows[2:]  # remaining tokenized lines: all upper- then lower-surface coordinate pairs

    upper = np.array([[float(x), float(y)] for x, y in coords[:n_upper]])  # upper-surface points, LE -> TE
    lower = np.array([[float(x), float(y)] for x, y in coords[n_upper:n_upper + n_lower]])  # lower-surface points, LE -> TE
    return upper, lower


def compute_tangents(pts):
    """Unit tangent vectors via natural cubic spline (arc-length parameterised)."""
    pts = np.asarray(pts, dtype=float)  # input points coerced to a float ndarray for numeric ops
    ds = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))  # Euclidean distance between consecutive points (segment lengths)
    s = np.concatenate([[0.0], np.cumsum(ds)])  # cumulative arc length at each point, used as the spline parameter
    cs_x = CubicSpline(s, pts[:, 0], bc_type='natural')  # natural cubic spline fitting x as a function of arc length
    cs_y = CubicSpline(s, pts[:, 1], bc_type='natural')  # natural cubic spline fitting y as a function of arc length
    dx = cs_x(s, 1)  # first derivative of x w.r.t. arc length at each point (tangent x-component)
    dy = cs_y(s, 1)  # first derivative of y w.r.t. arc length at each point (tangent y-component)
    T = np.column_stack([dx, dy])  # raw (unnormalised) tangent vectors at each point
    norms = np.linalg.norm(T, axis=1, keepdims=True)  # magnitude of each tangent vector, needed to normalise to unit length
    norms = np.where(norms < 1e-14, 1.0, norms)  # guard against divide-by-zero for degenerate (near-zero) tangents
    return T / norms


def quadratic_control_pts(pts):
    """
    For each segment (pts[i], pts[i+1]), find the quadratic Bezier control
    point as the intersection of the tangent lines at the two endpoints.

    Solves: lam*T0 + mu*T1 = P1 - P0  =>  C = P0 + lam*T0
    Falls back to chord midpoint when tangents are parallel or intersection
    lies behind an endpoint (lam<=0 or mu<=0).
    """
    pts = np.asarray(pts, dtype=float)  # input points coerced to a float ndarray for numeric ops
    T = compute_tangents(pts)  # unit tangent vector at each point, used to build the tangent-line intersection system
    ctrls = []  # accumulator for the resulting Bezier control point of each segment
    for i in range(len(pts) - 1):
        P0, P1 = pts[i], pts[i + 1]  # segment endpoints
        t0, t1 = T[i], T[i + 1]  # tangent directions at the two endpoints
        A = np.column_stack([t0, t1])  # 2x2 system matrix: columns are the two tangent directions
        b = P1 - P0  # right-hand side: vector between endpoints, used to solve lam*t0 + mu*t1 = b
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]  # determinant of A, used to detect parallel/degenerate tangents
        if abs(det) > 1e-12:
            lam, mu = np.linalg.solve(A, b)  # solve for the scalar steps along each tangent to their intersection
            if lam > 0 and mu > 0:
                C = P0 + lam * t0  # candidate control point: intersection of the two tangent lines
                seg = P1 - P0  # chord vector from P0 to P1, used to validate C's position along the segment
                alpha = np.dot(C - P0, seg) / (np.dot(seg, seg) + 1e-30)  # C's projected fraction along the chord (0=P0, 1=P1)
                if not (0.05 < alpha < 0.95):
                    C = (P0 + P1) / 2  # fallback: intersection too close to an endpoint, use chord midpoint instead
            else:
                C = (P0 + P1) / 2  # fallback: intersection lies behind an endpoint, use chord midpoint instead
        else:
            C = (P0 + P1) / 2  # fallback: tangents are parallel (no valid intersection), use chord midpoint instead
        ctrls.append(C)
    return np.array(ctrls)


def build_airfoil_segments(filepath):
    """
    Read a Lednicer-style airfoil coordinate file and build the quadratic
    Bezier segments describing the airfoil boundary.

    Path: lower TE -> LE -> upper TE (lower surface reversed), so segment i
    runs from airfoil_pts[i] to airfoil_pts[i+1] via control point ctrls[i].

    Returns
    -------
    segments : list of (P0, Pctrl, P1)
        One (start, control, end) triple per quadratic Bezier segment,
        each point given as [x, y].
    """
    upper, lower = read_lednicer_airfoil(filepath)  # raw upper/lower surface points read from the airfoil file

    lower_rev   = lower[::-1]  # lower surface reversed (TE -> LE) so it can be chained with the upper surface
    airfoil_pts = np.vstack([lower_rev, upper[1:]])  # full boundary path: lower TE -> LE -> upper TE (LE point deduplicated)
    ctrls       = quadratic_control_pts(airfoil_pts)  # Bezier control point for each consecutive segment of airfoil_pts

    return [(airfoil_pts[i].tolist(), ctrls[i].tolist(), airfoil_pts[i + 1].tolist())
            for i in range(len(airfoil_pts) - 1)]


def build_farfield_boundary(n, r):
    """
    Vertices of a regular n-gon (circumradius r, centred at the origin),
    used as the farfield boundary. As n grows the polygon tends to a circle.
    Vertices start at (r, 0) and proceed counter-clockwise.
    """
    theta = 2 * np.pi * np.arange(n) / n  # equally spaced angles around the circle, one per polygon vertex
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)]).tolist()


def write_in2d(airfoil_segments, farfield_pts, output_file):
    """
    Combine airfoil quadratic Bezier segments (bc=2) and a closed farfield
    polygon (bc=1) into a splinecurves2dv2 .in2d file.

    Parameters
    ----------
    airfoil_segments : list of (P0, Pctrl, P1)
        Output of build_airfoil_segments. The last segment's P1 is closed
        back to the first segment's P0 (sharp/blunt trailing edge).
    farfield_pts : list of [x, y]
        Output of build_farfield_boundary.
    output_file : str
        Path to write the new .in2d file.
    """
    n_seg = len(airfoil_segments)  # number of airfoil Bezier segments, used to index/loop and to size the TE wraparound

    # Interleave: P0 C0 P1 C1 P2 ... Pn
    point_list = []  # flat list of all output points (airfoil P0/control points, then farfield vertices), in file write order
    for P0, Pctrl, _ in airfoil_segments:
        point_list.append(P0)
        point_list.append(Pctrl)

    n_ff          = len(farfield_pts)  # number of farfield polygon vertices, used to loop and to wrap the closing edge
    outer_start_0 = len(point_list)      # 0-based offset for farfield pts
    for xy in farfield_pts:
        point_list.append(list(xy))

    # Build in2d lines
    out_lines = ["splinecurves2dv2", "1", "", "points"]  # accumulator for every line of the output .in2d file
    for idx, (x, y) in enumerate(point_list):
        out_lines.append(f"{idx + 1}\t{x:.15g}\t{y:.15g}")

    out_lines += ["", "segments"]

    for i in range(n_seg):
        p1, p2 = 2 * i + 1, 2 * i + 2  # 1-based point-list indices of this segment's start point and control point
        p3 = 2 * i + 3 if i < n_seg - 1 else 1  # sharp TE: close back to point 1
        out_lines.append(f"1\t0\t3\t{p1}\t{p2}\t{p3}\t-bc=2")

    for i in range(n_ff):
        p1 = outer_start_0 + i + 1  # 1-based point-list index of this farfield vertex
        p2 = outer_start_0 + (i + 1) % n_ff + 1  # 1-based index of the next vertex, wrapping to close the polygon
        out_lines.append(f"1\t0\t2\t{p1}\t{p2}\t-bc=1")

    out_lines += ["", "materials", "1\tdomain1"]

    with open(output_file, "w") as fout:
        fout.write("\n".join(out_lines))

    print(f"Airfoil on-curve points : {n_seg + 1}")
    print(f"Airfoil segments        : {n_seg} quadratic + 1 TE straight")
    print(f"Farfield points\\segs    : {n_ff}")
    print(f"Total points in file    : {len(point_list)}")
    print(f"Written -> {output_file}")
    return point_list


airfoil_segments = build_airfoil_segments(airfoil_file)  # quadratic Bezier (P0, Pctrl, P1) triples describing the airfoil boundary
farfield_pts     = build_farfield_boundary(n_farfield, r_farfield)  # n-gon vertices forming the farfield boundary
write_in2d(airfoil_segments, farfield_pts, output_file)