"""
uniform_refine.py
=================
Standalone uniform triangle refinement for Netgen .vol meshes.

Usage (CLI)
-----------
    python uniform_refine.py INPUT.vol OUTPUT.vol [--in2d GEOM.in2d] [--bl-bc 2] [--levels 1]

Usage (API)
-----------
    from uniform_refine import uniform_refine_mesh
    uniform_refine_mesh("mesh.vol", "refined.vol", in2d_file="geom.in2d", bl_bc=2, n_levels=2)

Behaviour
---------
- Only triangles are refined (1->4 red refinement).
- Quad elements pass through unchanged.
- Edges on the curved boundary layer (bc==bl_bc, described by 3-point Bezier
  segments in the .in2d file) are split at their rational quadratic parametric
  midpoint so the refined mesh stays on the true geometry.
- All other edges use the arithmetic midpoint.
- Boundary dist values are linearly interpolated at midpoints.
- The Bezier edge map is rebuilt after each refinement level so sub-edges at
  deeper levels are also curved.
"""

import argparse
import numpy as np
from typing import List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_mesh(mesh_file: str, metric_file: Optional[str] = None) -> Union[
        Tuple[List[List[int]], List[List[int]], List[List[float]]],
        Tuple[List[List[int]], List[List[int]], List[List[float]], List[List[float]]]
    ]:
    """
    Read a 2D triangular mesh file.

    Parameters
    ----------
    mesh_file : str
        Path to mesh file.
    metric_file : Optional[str]
        Path to metric file (optional). If provided, metric values
        are read and returned.

    Returns
    -------
    element_vertex_id : List[List[int]]
        element_vertex_id[k] = [n0, n1, n2]
        Triangle k made of vertices n0, n1, n2 (0-based indexing)

    boundary_data : {}
        boundary_data[i] = []
        Boundary edge with surface ID and vertex indices (0-based)

    vertex_coords : List[List[float]]
        vertex_coords[i] = [x, y]
        Coordinates of vertex i

    metric_values : List[List[float]], optional
        metric_values[i] = [m11, m12, m22] (or appropriate components)
        Metric tensor components at vertex i.
        Returned only if metric_file is provided.
    """

    with open(mesh_file, 'r') as f:
        lines = f.readlines()

    def find_section(keyword: str) -> int:
        """
        Return the line index of `keyword` in the mesh file.

        Parameters
        ----------
        keyword : str
            Section header string (e.g. 'surfaceelements').

        Returns
        -------
        int
            Zero-based index of the line that reads `keyword\n`.

        Raises
        ------
        ValueError
            If the keyword is not found in the file.
        """
        try:
            return lines.index(keyword + '\n')
        except ValueError:
            raise ValueError(f"Keyword '{keyword}' not found in mesh file")

    # --- surface elements ---
    idx = find_section('surfaceelements')
    nelm = int(lines[idx + 1].strip())

    element_vertex_id = []
    for i in range(nelm):
        tokens = lines[idx + 2 + i].split()
        np_elm = int(tokens[4])  # tokens: surfnr bcnr domin domout np_elm n0 n1 ...
        if len(tokens) < 5 + np_elm:
            raise ValueError(f"Invalid surface element line at {idx + 2 + i}")
        if np_elm == 3:
            element_vertex_id.append([int(tokens[5]) - 1, int(tokens[6]) - 1, int(tokens[7]) - 1])
        elif np_elm == 4:
            element_vertex_id.append([int(tokens[5]) - 1, int(tokens[6]) - 1, int(tokens[7]) - 1, int(tokens[8]) - 1])
        else:
            raise ValueError(f"Unsupported element with np={np_elm}")

    # --- boundary edges ---
    idx = find_section('edgesegmentsgi2')
    nbdry = int(lines[idx + 1].strip())

    boundary_data = {}
    for i in range(nbdry):
        tokens = lines[idx + 2 + i].split()
        parsed = []
        for val in tokens:
            try:
                parsed.append(int(val))
            except ValueError:
                parsed.append(float(val))
        boundary_data[i] = parsed
        boundary_data[i][2] -= 1  # convert 1-based node indices to 0-based
        boundary_data[i][3] -= 1

    # --- vertex coordinates ---
    idx = find_section('points')
    nvert = int(lines[idx + 1].strip())

    vertex_coords = []
    for i in range(nvert):
        tokens = lines[idx + 2 + i].split()
        vertex_coords.append([float(tokens[0]), float(tokens[1])])

    # --- optional metric file ---
    if metric_file is not None:
        metric_values = []
        with open(metric_file, 'r') as f:
            next(f)
            for line in f:
                tokens = line.split()
                metric_values.append([float(val) for val in tokens])
        return element_vertex_id, boundary_data, vertex_coords, metric_values

    return element_vertex_id, boundary_data, vertex_coords


def exportMesh1(newfile, element_vertex_id, ordered_boundary_data, vertices):
    """
    Write a hybrid mesh to a Netgen .vol file.

    Triangles are written before quads in the surfaceelements section as
    required by Netgen.  All node indices are converted to 1-based on output.

    Parameters
    ----------
    newfile : str
        Output file path.
    element_vertex_id : list[list[int]]
        Element connectivity (mix of triangles and quads), 0-based indices.
    ordered_boundary_data : dict
        Boundary-edge records as produced by order_boundary_data or read_mesh,
        with 0-based node indices.
    vertices : list[list[float]]
        Vertex coordinates [x, y].
    """

    nelm_total = len(element_vertex_id)
    nvert = len(vertices)
    bedges = len(ordered_boundary_data)

    with open(newfile, 'w') as f:

        f.write("mesh3d\ndimension\n2\ngeomtype\n0\n")
        f.write('# surfnr    bcnr    domin    domout    np      p1      p2      p3\n')
        f.write('surfaceelements\n')
        f.write(str(nelm_total) + "\n")

        # Netgen requires triangles listed before quads in surfaceelements
        for elm in element_vertex_id:
            if len(elm) == 3:
                f.write(
                    f"{2:>8}{1:>8}{0:>8}{0:>8}{3:>8}"
                    f"{elm[0]+1:>8}{elm[1]+1:>8}{elm[2]+1:>8}\n"
                )
        for elm in element_vertex_id:
            if len(elm) == 4:
                f.write(
                    f"{2:>8}{1:>8}{0:>8}{0:>8}{4:>8}"
                    f"{elm[0]+1:>8}{elm[1]+1:>8}{elm[2]+1:>8}{elm[3]+1:>8}\n"
                )

        f.write('\n# matnr    np    p1    p2    p3    p4\n')
        f.write('volumeelements\n0\n\n')

        f.write('# surfid       0      p1       p2   trinum1  trinum2 domin/sf1 domout/sf2 ednr1    dist1     ednr2     dist2\n')
        f.write('edgesegmentsgi2\n')
        f.write(str(bedges) + "\n")

        for k in range(bedges):
            val = ordered_boundary_data[k]
            f.write(
                f"{val[0]}\t{0}\t{val[2]+1}\t{val[3]+1}\t"
                f"{-1}\t{-1}\t{1}\t{0}\t"
                f"{val[8]}\t{val[9]:.16e}\t{val[10]}\t{val[11]:.16e}\n"
            )

        f.write('#     X     Y     Z\n')
        f.write('points\n')
        f.write(str(nvert) + "\n")

        for v in vertices:
            f.write(f"{v[0]:>24.16e}{v[1]:>29.16e}{0:>29.16e}\n")

        f.write('materials\n1\n1 domain1\n\nendmesh')


# ---------------------------------------------------------------------------
# .in2d geometry parser
# ---------------------------------------------------------------------------

def parse_in2d_spline_segments(in2d_file, bc_flag=2):
    """
    Parse rational quadratic Bezier (conic) segments from an .in2d splinecurves2dv2 file.

    Each 3-point wall segment has p0 and p2 as curve endpoints and p_ctrl as
    the tangent intersection control point (NOT on the curve).

    Returns list of (p0, p_ctrl, p2) numpy arrays.
    """
    with open(in2d_file, encoding="utf-8") as f:
        lines = f.readlines()

    pts = {}
    in_pts = False
    for line in lines:
        s = line.strip()
        if s == "points":
            in_pts = True
            continue
        if s in ("segments", "materials"):
            in_pts = False
            continue
        if in_pts and s:
            tokens = s.split()
            if len(tokens) >= 3:
                try:
                    pts[int(tokens[0])] = np.array([float(tokens[1]), float(tokens[2])])
                except ValueError:
                    pass

    segments = []
    in_segs = False
    for line in lines:
        s = line.strip()
        if s == "segments":
            in_segs = True
            continue
        if s == "materials":
            break
        if in_segs and "-bc=" in s:
            tokens = s.split()
            bc_tok = [t for t in tokens if t.startswith("-bc=")]
            if bc_tok and int(bc_tok[0].split("=")[1]) == bc_flag:
                if int(tokens[2]) == 3:
                    segments.append((pts[int(tokens[3])], pts[int(tokens[4])], pts[int(tokens[5])]))

    return segments


# ---------------------------------------------------------------------------
# Uniform refinement
# ---------------------------------------------------------------------------

# ============================================================
# Uniform triangle refinement with Bezier-aware boundary midpoints
# ============================================================
# Each triangle is split 1->4 by inserting edge midpoints.
# Edges that lie on the quadratic Bezier boundary layer use the
# rational quadratic parametric form (formula A.3) to place the
# midpoint ON the curve.  All other edges use the arithmetic midpoint.
#
# Rational quadratic Bezier (weight w derived from control points):
#   B(t) = [ (1-t)^2 P0 + w*t*(1-t) P_ctrl + t^2 P2 ]
#          / [ (1-t)^2 + w*t*(1-t)        + t^2      ]
#
# Weight:  w = ||P0 - P2|| / sqrt(0.5*(||P0-P_ctrl||^2 + ||P2-P_ctrl||^2))
# ============================================================

def bezier_point(p0, p_ctrl, p2, t):
    """Evaluate rational quadratic Bezier (formula A.3) at parameter t."""
    w_num = np.linalg.norm(p0 - p2)
    w_den = np.sqrt(0.5 * (np.linalg.norm(p0 - p_ctrl)**2 + np.linalg.norm(p2 - p_ctrl)**2))
    w     = w_num / w_den if w_den > 1e-14 else 1.0
    denom = (1 - t)**2 + w * t * (1 - t) + t**2
    return ((1 - t)**2 * p0 + w * t * (1 - t) * p_ctrl + t**2 * p2) / denom


def build_bl_edge_map(spline_segs, boundary_data, bl_bc):
    """
    Match boundary edges (bc == bl_bc) to rational quadratic Bezier segments
    using ednr/dist fields from the .vol boundary records.

    Returns dict  (min_n, max_n) -> (p0, p_ctrl, p2, dist_a, dist_b)
    where dist_a is the parametric position at key[0] and dist_b at key[1].
    Edges that straddle two geometry segments (ednr1 != ednr2) are skipped.
    """
    bl_edge_map = {}
    for val in boundary_data.values():
        if val[0] != bl_bc:
            continue
        n1, n2 = val[2], val[3]
        ednr1, dist1 = int(val[8]),  float(val[9])
        ednr2, dist2 = int(val[10]), float(val[11])
        if ednr1 != ednr2:
            continue
        si = ednr1 - 1
        if si < 0 or si >= len(spline_segs):
            continue
        p0, p_ctrl, p2 = spline_segs[si]
        key = (min(n1, n2), max(n1, n2))
        if key[0] == n1:
            bl_edge_map[key] = (p0, p_ctrl, p2, dist1, dist2)
        else:
            bl_edge_map[key] = (p0, p_ctrl, p2, dist2, dist1)
    return bl_edge_map


def refine_triangles(vertex_coords, triangles, boundary_data,
                     bezier_edge_map=None):
    """
    Uniform 1->4 triangle refinement with Bezier-aware edge midpoints.

    Each triangle is split into 4 children by inserting midpoints on its
    three edges.  For edges recorded in bezier_edge_map (edges lying on
    the quadratic Bezier boundary layer) the midpoint is placed on the
    curve using the rational parametric form.  All other edges use the
    arithmetic midpoint.

    Child layout (i0, i1, i2 = original corners; m01, m12, m02 = midpoints):

        [i0,  m01, m02]   corner at i0
        [m01, i1,  m12]   corner at i1
        [m02, m12, i2 ]   corner at i2
        [m01, m12, m02]   centre

    Boundary edges are split at their midpoints.  The dist values in the
    boundary records are linearly interpolated (theta = 0.5 at the
    midpoint) so NGSolve maps each sub-edge to the correct position on
    the geometry segment.

    Parameters
    ----------
    vertex_coords   : list of [x, y]
    triangles       : list of [n0, n1, n2] (0-based)
    boundary_data   : dict as returned by read_mesh (not mutated in-place)
    bezier_edge_map : dict from build_bl_edge_map, or None

    Returns
    -------
    new_vertex_coords : list of [x, y]
    new_triangles     : list of [n0, n1, n2]
    new_boundary_data : dict with split boundary edge records
    """
    nodes           = np.array(vertex_coords, dtype=float)
    bezier_edge_map = bezier_edge_map or {}
    new_nodes       = [v[:] for v in vertex_coords]
    edge_to_mid     = {}   # canonical key -> new node index

    def _get_mid(i, j):
        key = (min(i, j), max(i, j))
        if key in edge_to_mid:
            return edge_to_mid[key]

        if key in bezier_edge_map:
            p0, p_ctrl, p2, dist_a, dist_b = bezier_edge_map[key]
            t_mid = (dist_a + dist_b) * 0.5
            mid   = bezier_point(p0, p_ctrl, p2, t_mid).tolist()
        else:
            mid = ((nodes[i] + nodes[j]) * 0.5).tolist()

        mid_idx = len(new_nodes)
        new_nodes.append(mid)
        edge_to_mid[key] = mid_idx
        return mid_idx

    new_tris = []
    for i0, i1, i2 in triangles:
        m01 = _get_mid(i0, i1)
        m12 = _get_mid(i1, i2)
        m02 = _get_mid(i0, i2)
        new_tris.extend([
            [i0,  m01, m02],
            [m01, i1,  m12],
            [m02, m12, i2 ],
            [m01, m12, m02],
        ])

    # ----------------------------------------------------------------
    # Update boundary data: split every boundary edge that was refined.
    # dist is linearly interpolated; theta = 0.5 always at the midpoint.
    # ----------------------------------------------------------------
    new_boundary = {}
    bdry_idx = 0
    for val in boundary_data.values():
        n2, n3 = val[2], val[3]
        key    = (min(n2, n3), max(n2, n3))

        if key not in edge_to_mid:
            new_boundary[bdry_idx] = val[:]
            bdry_idx += 1
            continue

        mid_idx = edge_to_mid[key]
        d_start = val[9]
        d_end   = val[11]
        d_mid   = (d_start + d_end) * 0.5

        sub_a       = val[:]
        sub_b       = val[:]
        sub_a[2],  sub_a[3]  = n2,      mid_idx
        sub_a[9],  sub_a[11] = d_start, d_mid
        sub_b[2],  sub_b[3]  = mid_idx, n3
        sub_b[9],  sub_b[11] = d_mid,   d_end

        new_boundary[bdry_idx] = sub_a; bdry_idx += 1
        new_boundary[bdry_idx] = sub_b; bdry_idx += 1

    return new_nodes, new_tris, new_boundary


def uniform_refine_mesh(mesh_file, output_file,
                        in2d_file=None, bl_bc=2,
                        n_levels=1, tol=1e-8):
    """
    Uniformly refine the triangles of a mesh and write the result.

    Each refinement level splits every triangle into 4 children (1->4).
    For triangles that have an edge on the Bezier boundary layer (bc==bl_bc,
    described by 3-point segments in in2d_file), the midpoint of that edge
    is placed on the rational quadratic Bezier curve rather than on the
    chord, keeping the refined mesh on the true geometry.

    Quad elements are passed through unchanged; only triangles are refined.
    Boundary edge records are split and dist values are linearly interpolated.

    Parameters
    ----------
    mesh_file   : str   input .vol file
    output_file : str   output .vol file
    in2d_file   : str or None
                  Netgen .in2d file with 3-point Bezier wall segments.
                  Pass None to use arithmetic midpoints for all edges.
    bl_bc       : int   BC flag of the curved boundary (default 2)
    n_levels    : int   number of refinement levels (default 1)
    tol         : float geometric tolerance for on-curve detection

    Returns
    -------
    elements      : list of element connectivity (tris then quads)
    boundary_data : updated boundary dict
    vertex_coords : updated coordinate list

    Example
    -------
    uniform_refine_mesh("temp.vol", "temp_refined.vol",
                        in2d_file="netgen.in2d", bl_bc=2, n_levels=2)
    """
    elements, boundary_data, vertex_coords = read_mesh(mesh_file)

    triangles = [e for e in elements if len(e) == 3]
    quads     = [e for e in elements if len(e) == 4]

    # Build Bezier edge map from the in2d file for the curved boundary
    bezier_edge_map = {}
    spline_segs     = []
    if in2d_file is not None:
        spline_segs = parse_in2d_spline_segments(in2d_file, bc_flag=bl_bc)
        if spline_segs:
            bezier_edge_map = build_bl_edge_map(spline_segs, boundary_data, bl_bc)

    for _level in range(n_levels):
        vertex_coords, triangles, boundary_data = refine_triangles(
            vertex_coords, triangles, boundary_data, bezier_edge_map
        )
        # After each level the boundary edges have been split; rebuild the
        # map so that the new sub-edges on the Bezier are also curved.
        if spline_segs:
            bezier_edge_map = build_bl_edge_map(spline_segs, boundary_data, bl_bc)

    all_elements = triangles + quads
    exportMesh1(output_file, all_elements, boundary_data, vertex_coords)

    print("------------------------------------------------")
    print("Uniform triangle refinement complete")
    print(f"  Refinement levels : {n_levels}")
    print(f"  Triangles         : {len(triangles)}")
    print(f"  Quads (unchanged) : {len(quads)}")
    print(f"  Total elements    : {len(all_elements)}")
    print(f"  Total nodes       : {len(vertex_coords)}")
    if bezier_edge_map:
        print(f"  Bezier BL edges   : {len(bezier_edge_map)} (bc={bl_bc})")
    else:
        print("  No Bezier segments found; arithmetic midpoints used for all edges")
    print("------------------------------------------------")

    return all_elements, boundary_data, vertex_coords


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Uniformly refine the triangle elements of a Netgen .vol mesh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mesh_file",   help="Input .vol mesh file")
    p.add_argument("output_file", help="Output .vol mesh file")
    p.add_argument("--in2d",   dest="in2d_file",  default=None,
                   help="Netgen .in2d file with 3-point Bezier wall segments")
    p.add_argument("--bl-bc",  dest="bl_bc",      type=int, default=2,
                   help="BC flag of the curved boundary layer")
    p.add_argument("--levels", dest="n_levels",   type=int, default=1,
                   help="Number of refinement levels")
    p.add_argument("--tol",    dest="tol",        type=float, default=1e-8,
                   help="Geometric tolerance for on-curve detection")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    uniform_refine_mesh(
        mesh_file   = args.mesh_file,
        output_file = args.output_file,
        in2d_file   = args.in2d_file,
        bl_bc       = args.bl_bc,
        n_levels    = args.n_levels,
        tol         = args.tol,
    )
