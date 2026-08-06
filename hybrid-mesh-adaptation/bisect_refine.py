"""
Adaptive mesh refinement via newest-vertex bisection.

Triangle convention
-------------------
Each row of `elem` is  [newest_vertex,  base_v1,  base_v2]  (0-based indices).
The "base edge" is the edge between base_v1 and base_v2 — the edge opposite
the newest vertex and the ONLY edge ever split.

Typical adaptive loop
---------------------
    from bisect_refine import load_vol, load_eta, bisect_mesh, export_mesh

    coord, elem, seg_data = load_vol("netgen_tri330.vol")
    eta = load_eta("cellwise_error_330.csv")

    for _ in range(max_iters):
        coord, elem, seg_data = bisect_mesh(coord, elem, eta, theta=0.5,
                                            boundary=seg_data)
        export_mesh("refined.vol", coord, elem, seg_data)
        # ... recompute eta on the new mesh ...

seg_data format (matches uniform_refine.py's boundary_data)
------------------------------------------------------------
    seg_data = {
        0: [surfid, 0, p1, p2, -1, -1, 1, 0, ednr1, dist1, ednr2, dist2],
        1: [...],
        ...
    }
where p1, p2 are 0-based node indices.
"""

import numpy as np
from uniform_refine import (read_mesh, exportMesh1,
                             parse_in2d_spline_segments,
                             build_bl_edge_map,
                             bezier_point)


# ---------------------------------------------------------------------------
# .vol mesh loader
# ---------------------------------------------------------------------------

def load_vol(filepath):
    """
    Load a 2-D triangular mesh from a Netgen .vol file.

    Uses read_mesh (from deformation_aflr / uniform_refine) to parse all three
    sections, then reorders each triangle so that the longest edge becomes the
    base edge (columns 1-2) and its opposite vertex is the newest vertex
    (column 0) — the required starting convention for newest-vertex bisection.

    Parameters
    ----------
    filepath : str
        Path to the .vol file.

    Returns
    -------
    coord    : ndarray, shape (n_nodes, 2)
        Node (x, y) coordinates.
    elem     : ndarray, shape (n_elem, 3)
        Triangle connectivity, 0-based, format [newest_vertex, base_v1, base_v2].
    seg_data : dict  {i: list_of_12_fields}
        Boundary edge records as returned by read_mesh; p1/p2 are 0-based.
        Compatible with exportMesh1 and uniform_refine.py's boundary_data format.
    """
    # elem_raw: raw element connectivity list from the .vol file, before reordering
    # seg_data: dict of boundary edge records keyed by sequential index
    # vertex_coords: list of [x, y] coordinates for every node
    elem_raw, seg_data, vertex_coords = read_mesh(filepath)

    # coord: numpy float array of node coordinates, shape (n_nodes, 2)
    coord    = np.array(vertex_coords, dtype=float)
    # elem_raw: converted to numpy int array for indexing into coord
    elem_raw = np.array(elem_raw,      dtype=int)
    # elem: reordered so each row is [newest_vertex, base_v1, base_v2]
    elem     = assign_newest_vertex(coord, elem_raw)

    return coord, elem, seg_data


def assign_newest_vertex(coord, elem):
    """
    Reorder each triangle so the longest edge is the base edge (columns 1-2)
    and the opposite vertex is the newest vertex (column 0).
    """
    # elem: working copy so the input array is not mutated
    elem = elem.copy()
    for t in range(len(elem)):      # t: triangle index
        v = elem[t]                 # v: node indices [v0, v1, v2] of triangle t
        p = coord[v]                # p: (3, 2) array of (x, y) coords for v0, v1, v2

        # l01, l12, l20: squared lengths of edges v0-v1, v1-v2, v2-v0
        l01 = np.sum((p[0] - p[1]) ** 2)
        l12 = np.sum((p[1] - p[2]) ** 2)
        l20 = np.sum((p[2] - p[0]) ** 2)

        # longest: 0, 1, or 2 — index of the longest edge among l01, l12, l20
        longest = np.argmax([l01, l12, l20])

        if longest == 0:               # longest: v0-v1  →  newest: v2
            elem[t] = [v[2], v[0], v[1]]
        elif longest == 1:             # longest: v1-v2  →  newest: v0  (already correct)
            pass
        else:                          # longest: v2-v0  →  newest: v1
            elem[t] = [v[1], v[2], v[0]]

    return elem


# ---------------------------------------------------------------------------
# CSV error loader
# ---------------------------------------------------------------------------

def load_eta(filepath):
    """
    Load per-element error indicators from a CSV file.

    Expected format (no header, UTF-8 or UTF-8-BOM):
        element_index , error_value , ...

    The element index in column 0 is used to order the output so rows do
    not need to be pre-sorted.  Extra columns beyond the first two are ignored.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (e.g. "cellwise_error_330.csv").

    Returns
    -------
    eta : ndarray, shape (n_elem,)
        Error value for each element, ordered by element index.
    """
    # indices: element indices read from column 0 of the CSV
    indices = []
    # errors: error values read from column 1 of the CSV
    errors  = []

    with open(filepath, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()     # line: current CSV row with whitespace removed
            if not line:
                continue
            parts = line.split(",") # parts: individual comma-separated fields of this row
            indices.append(int(parts[0]))   # column 0: element index
            errors.append(float(parts[1]))  # column 1: error value

    # n: size of the output array — one slot per element index, 0-based
    n = max(indices) + 1
    # eta: zero-initialised array; filled by index so unsorted CSV rows are handled correctly
    eta = np.zeros(n, dtype=float)
    for idx, err in zip(indices, errors):   # idx: element index, err: its error value
        eta[idx] = err

    return eta


# ---------------------------------------------------------------------------
# Data-structure helpers
# ---------------------------------------------------------------------------

def build_edge_data(coord, elem):
    """
    Build node-to-element and node-to-edge lookup tables.

    Parameters
    ----------
    coord : list of [x, y]
    elem  : list of [newest_vertex, base_v1, base_v2]

    Returns
    -------
    n2el    : dict (i, j) -> int   element index with directed edge i -> j
    n2ed    : dict (i, j) -> int   edge index for undirected edge {i, j}
    n_edges : int                  total number of unique edges
    """
    # n2el: directed edge (i→j) → index of the triangle that owns that directed edge
    n2el = {}
    for t, v in enumerate(elem):   # t: triangle index, v: its three node indices
        n2el[(v[0], v[1])] = t
        n2el[(v[1], v[2])] = t
        n2el[(v[2], v[0])] = t

    # unique_edges: canonical (min,max) edge key → unique integer edge id
    unique_edges = {}
    # idx: counter that assigns a fresh id to each new edge
    idx = 0
    for v in elem:
        for k in range(3):                          # k: local edge slot 0/1/2 within triangle
            a, b = v[k], v[(k + 1) % 3]            # a, b: endpoints of edge k
            key = (min(a, b), max(a, b))            # key: canonical undirected edge identifier
            if key not in unique_edges:
                unique_edges[key] = idx
                idx += 1

    # n2ed: undirected edge lookup — both (a,b) and (b,a) map to the same edge id
    n2ed = {}
    for (a, b), eid in unique_edges.items():        # eid: unique integer id for this edge
        n2ed[(a, b)] = eid
        n2ed[(b, a)] = eid

    return n2el, n2ed, idx


def divide(elem, t, p):
    """
    Bisect triangle t by inserting midpoint M on its base edge.

    p = [v1, v2, v3, M]

    After the call:
        elem[t]  = [M, v1, v2]   left child   (newest vertex = M, base = v1-v2)
        appended = [M, v3, v1]   right child  (newest vertex = M, base = v3-v1)
    """
    # v1: newest vertex of the parent triangle (kept as base endpoint in left child)
    # v2: first base endpoint (kept as base endpoint in left child)
    # v3: second base endpoint (kept as base endpoint in right child)
    # M: new midpoint node inserted on the base edge v2-v3
    v1, v2, v3, M = p
    elem.append([M, v3, v1])    # right child: newest=M, base edge = v3-v1
    elem[t] = [M, v1, v2]      # left child:  newest=M, base edge = v1-v2 (replaces parent)
    return elem


# ---------------------------------------------------------------------------
# Main refinement function
# ---------------------------------------------------------------------------

def pass1_mark(coord, elem, eta, theta, n2el, n2ed, bezier_edge_map):
    """
    Pass 1 — Dörfler marking + conformity propagation.

    Walk elements in descending error order. For each chosen element, mark its
    base edge (inserting a midpoint node) and follow the neighbor chain to
    prevent hanging nodes. Bezier edges get curve-accurate midpoints.

    Parameters
    ----------
    coord          : list of [x, y]  — mutated in place (midpoints appended)
    elem           : list of [newest, base1, base2]
    eta            : ndarray of per-element error indicators
    theta          : Dörfler fraction in (0, 1]
    n2el           : directed edge → triangle index
    n2ed           : undirected edge → unique edge id
    bezier_edge_map: (min,max) edge key → Bezier control data

    Returns
    -------
    marker : dict  edge_id → new midpoint node index
    """
    # total: sum of all error indicators; Dörfler threshold is theta * total
    total = float(eta.sum())
    # order: element indices sorted by descending error (largest error first)
    order = np.argsort(-eta)

    # marker: edge_id → new midpoint node index for every base edge that has been marked
    marker  = {}
    # current: cumulative error of all marked elements; loop stops when it exceeds theta*total
    current = 0.0

    for t_rank in order:               # t_rank: element index in descending-error order
        if current > theta * total:
            break

        # ct: current triangle being processed; walks the neighbor chain for conformity
        ct = int(t_rank)

        while True:
            v    = elem[ct]                     # v: [newest, base1, base2] of current triangle
            base = n2ed.get((v[1], v[2]))       # base: unique edge id of the base edge

            if base is None:                    # boundary edge with no entry in n2ed
                break
            if base in marker:                  # base edge already marked in a previous iteration
                break

            current     += eta[ct]              # accumulate error of this newly marked element
            N            = len(coord)           # N: index of the next node to be added
            marker[base] = N                    # record midpoint node index for this edge

            # key: canonical (min, max) edge key for Bezier lookup
            key = (min(v[1], v[2]), max(v[1], v[2]))
            if key in bezier_edge_map:
                # p0, p_ctrl, p2: Bezier control points of the curved BL edge
                # dist_a, dist_b: parametric distances at the two endpoints
                p0, p_ctrl, p2, dist_a, dist_b = bezier_edge_map[key]
                # mid: [x, y] of the midpoint placed exactly on the Bezier curve
                mid = bezier_point(p0, p_ctrl, p2, (dist_a + dist_b) / 2.0).tolist()
            else:
                # mid: [x, y] arithmetic midpoint for interior or straight edges
                mid = [(coord[v[1]][0] + coord[v[2]][0]) / 2.0,
                       (coord[v[1]][1] + coord[v[2]][1]) / 2.0]
            coord.append(mid)                   # append the new midpoint node

            # neighbor: triangle sharing the base edge from the opposite side
            neighbor = n2el.get((v[2], v[1]))
            if neighbor is None:                # base edge is on the mesh boundary
                break
            ct = neighbor                       # step to neighbor and continue the chain

    return marker


def pass2_bisect(elem, NT, n2ed, marker):
    """
    Pass 2 — Bisect all triangles whose base edge was marked.

    Iterates only over the NT original triangles. For each marked triangle,
    calls divide() to split it and immediately bisects any marked child edges.

    Parameters
    ----------
    elem   : list of [newest, base1, base2]  — mutated in place
    NT     : int  number of original triangles (before any bisection)
    n2ed   : undirected edge → unique edge id
    marker : dict  edge_id → midpoint node index

    Returns
    -------
    elem : updated list with bisected triangles
    """
    for t in range(NT):                     # t: triangle index (only original triangles)
        v    = elem[t]                      # v: [newest, base1, base2] of triangle t
        base = n2ed.get((v[1], v[2]))       # base: edge id of the base edge

        if base is None or base not in marker:
            continue

        # M: index of the midpoint node inserted on this triangle's base edge
        M = marker[base]
        # p: [newest, base1, base2, M] passed to divide() to produce the two children
        p = [v[0], v[1], v[2], M]

        elem = divide(elem, t, p)           # replace triangle t with left child; append right child

        # right_base: edge id of the right child's base edge (base2-newest)
        right_base = n2ed.get((p[2], p[0]))
        if right_base is not None and right_base in marker:
            # Mr: midpoint node for the right child's base edge (cascading bisection)
            Mr   = marker[right_base]
            elem = divide(elem, len(elem) - 1, [M, p[2], p[0], Mr])

        # left_base: edge id of the left child's base edge (newest-base1)
        left_base = n2ed.get((p[0], p[1]))
        if left_base is not None and left_base in marker:
            # Ml: midpoint node for the left child's base edge (cascading bisection)
            Ml   = marker[left_base]
            elem = divide(elem, t, [M, p[0], p[1], Ml])

    return elem


def pass3_boundary(boundary, bdry, is_seg_data, n2ed, marker):
    """
    Pass 3 — Update boundary edge data.

    Any boundary edge whose midpoint was inserted is split into two half-edges.
    For seg_data, parametric dist values are linearly interpolated at the midpoint.

    Parameters
    ----------
    boundary    : original boundary input (dict or array_like or None)
    bdry        : plain [p1, p2] list extracted from boundary
    is_seg_data : bool — True if boundary is a full seg_data dict
    n2ed        : undirected edge → unique edge id
    marker      : dict  edge_id → midpoint node index

    Returns
    -------
    boundary : updated boundary in the same type as the input
    """
    if is_seg_data:
        # boundary: updated seg_data dict with split edge records
        boundary = split_seg_data(boundary, n2ed, marker)
    else:
        bdry = split_boundary(bdry, n2ed, marker)
        # boundary: numpy array of [p1, p2] pairs, or empty (0,2) array if none remain
        boundary = np.array(bdry, dtype=int) if bdry else np.empty((0, 2), dtype=int)
    return boundary


def bisect_mesh(coord, elem, eta, theta=0.5, boundary=None,
                in2d_file=None, bl_bc=2):
    """
    Refine a triangular mesh by bisecting high-error elements.

    Orchestrates three passes:
      Pass 1 — Dörfler marking + conformity propagation  (pass1_mark)
      Pass 2 — Bisect all marked triangles               (pass2_bisect)
      Pass 3 — Update boundary edge records              (pass3_boundary)

    Parameters
    ----------
    coord : array_like, shape (n_nodes, 2)
        Node coordinates.
    elem : array_like, shape (n_elem, 3)
        Triangle connectivity.  Row format: [newest_vertex, base_v1, base_v2].
        Node indices are 0-based.
    eta : array_like, shape (n_elem,)
        Non-negative error indicator per element, already computed externally.
    theta : float, optional
        Dörfler parameter in (0, 1].  Default 0.5.
    boundary : dict or array_like, optional
        Boundary edge data (seg_data dict or plain (n,2) node-pair array).
        Pass None if there are no boundary edges to update.
    in2d_file : str or None, optional
        Path to the Netgen .in2d geometry file for Bezier-accurate midpoints.
    bl_bc : int, optional
        BC flag identifying the curved boundary layer in the .in2d file.

    Returns
    -------
    coord    : ndarray, shape (n_nodes_new, 2)
    elem     : ndarray, shape (n_elem_new, 3)
    boundary : same type as input
    """
    # coord: converted to a plain Python list so new midpoints can be appended
    coord = [list(map(float, c)) for c in coord]
    # elem: converted to a plain Python list for in-place updates
    elem  = [list(map(int,   e)) for e in elem]
    # eta: numpy array of per-element error indicators
    eta   = np.asarray(eta, dtype=float)

    # is_seg_data: True when boundary is a full seg_data dict (12-field records)
    is_seg_data = isinstance(boundary, dict)

    if is_seg_data:
        # bdry: plain [p1, p2] pairs extracted from seg_data for pass3
        bdry = [[val[2], val[3]] for val in boundary.values()]
    elif boundary is not None:
        # bdry: plain [p1, p2] pairs converted from array_like input
        bdry = [list(map(int, e)) for e in boundary]
    else:
        bdry = []

    # bezier_edge_map: (min_node, max_node) → Bezier params for curved BL edges
    bezier_edge_map = {}
    if in2d_file is not None and is_seg_data:
        # spline_segs: Bezier triples parsed from the .in2d file
        spline_segs = parse_in2d_spline_segments(in2d_file, bc_flag=bl_bc)
        if spline_segs:
            bezier_edge_map = build_bl_edge_map(spline_segs, boundary, bl_bc)

    # n2el: directed edge → triangle index
    # n2ed: undirected edge → unique edge id
    n2el, n2ed, _ = build_edge_data(coord, elem)
    # NT: number of original triangles; pass2 only iterates over these
    NT = len(elem)

    marker = pass1_mark(coord, elem, eta, theta, n2el, n2ed, bezier_edge_map)
    elem   = pass2_bisect(elem, NT, n2ed, marker)
    boundary = pass3_boundary(boundary, bdry, is_seg_data, n2ed, marker)

    return (
        np.array(coord, dtype=float),
        np.array(elem,  dtype=int),
        boundary,
    )


def split_seg_data(seg_data, n2ed, marker):
    """
    Split boundary edge records at marked midpoints, interpolating dist values.
    """
    # new_data: output dict of updated boundary edge records
    new_data = {}
    # new_idx: sequential integer key for entries in new_data
    new_idx  = 0

    for val in seg_data.values():       # val: one 12-field boundary edge record
        p1, p2 = val[2], val[3]         # p1, p2: start and end node indices of this edge
        eid    = n2ed.get((p1, p2))     # eid: unique edge id for the p1-p2 edge

        if eid is not None and eid in marker:
            M     = marker[eid]                     # M: midpoint node index inserted on this edge
            d_mid = (val[9] + val[11]) / 2.0        # d_mid: parametric dist at the midpoint (average of p1 and p2 dists)

            # sub_a: copy of val for the first half-edge p1 → M
            sub_a     = val[:]
            sub_a[3]  = M               # update end node to midpoint
            sub_a[10] = val[8]          # ednr for M = same curve segment as p1
            sub_a[11] = d_mid           # dist at the new end node (M)

            # sub_b: copy of val for the second half-edge M → p2
            sub_b    = val[:]
            sub_b[2] = M               # update start node to midpoint
            sub_b[8] = val[8]          # ednr for M = same curve segment
            sub_b[9] = d_mid           # dist at the new start node (M)

            new_data[new_idx] = sub_a;  new_idx += 1
            new_data[new_idx] = sub_b;  new_idx += 1
        else:
            new_data[new_idx] = val[:]  # edge not split — copy record unchanged
            new_idx += 1

    return new_data


def split_boundary(edges, n2ed, marker):
    """Split plain [p1, p2] boundary edge list at marked midpoints."""
    # extras: new second half-edges collected during the loop, appended afterwards
    extras = []
    for i, e in enumerate(edges):       # i: position in list, e: [p1, p2] edge
        eid = n2ed.get((e[0], e[1]))    # eid: unique edge id for this boundary edge
        if eid is not None and eid in marker:
            M = marker[eid]             # M: midpoint node inserted on this edge
            extras.append([M, e[1]])    # second half-edge: M → p2
            edges[i] = [e[0], M]        # replace original with first half-edge: p1 → M
    edges.extend(extras)
    return edges


# ---------------------------------------------------------------------------
# .vol mesh exporter
# ---------------------------------------------------------------------------

def export_mesh(filepath, coord, elem, seg_data):
    """
    Write a 2-D triangular mesh to a Netgen .vol file.

    Delegates to exportMesh1 (from deformation_aflr / uniform_refine) after
    converting numpy arrays to plain Python lists.

    Parameters
    ----------
    filepath : str
        Output file path (e.g. "refined.vol").
    coord : array_like, shape (n_nodes, 2)
        Node (x, y) coordinates.
    elem : array_like, shape (n_elem, 3)
        Triangle connectivity, 0-based.
    seg_data : dict
        Boundary edge records as returned by load_vol or bisect_mesh.
        Each value is a 12-field list:
          [surfid, 0, p1, p2, trinum1, trinum2, domin, domout, ednr1, dist1, ednr2, dist2]
        with p1, p2 as 0-based node indices.
    """
    exportMesh1(
        filepath,
        [list(map(int, e)) for e in elem],
        seg_data,
        [list(map(float, c)) for c in coord],
    )


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    # base: directory containing this script — used to build absolute paths to data files
    base = os.path.dirname(__file__)

    # vol_path: input .vol mesh file for this refinement step
    vol_path = os.path.join(base, "refined_bisect289_3.vol")
    # csv_path: CSV file with per-element error indicators for the input mesh
    csv_path = os.path.join(base, "cellwise_error_690.csv")
    # out_path: output .vol file for the refined mesh
    out_path = os.path.join(base, "refined_bisect289_4.vol")

    if not (os.path.exists(vol_path) and os.path.exists(csv_path)):
        print("netgen_tri330.vol or cellwise_error_330.csv not found — skipping example")
    else:
        # Load mesh and errors
        coord, elem, seg_data = load_vol(vol_path)
        eta = load_eta(csv_path)

        print("Original mesh:")
        print(f"  nodes: {len(coord)},  elements: {len(elem)},  boundary edges: {len(seg_data)}")

        # One refinement step
        coord, elem, seg_data = bisect_mesh(coord, elem, eta, theta=0.5, boundary=seg_data)

        print("\nAfter one refinement step (theta=0.5):")
        print(f"  nodes: {len(coord)},  elements: {len(elem)},  boundary edges: {len(seg_data)}")

        # Export refined mesh
        export_mesh(out_path, coord, elem, seg_data)
        print(f"\nExported refined mesh -> {out_path}")
