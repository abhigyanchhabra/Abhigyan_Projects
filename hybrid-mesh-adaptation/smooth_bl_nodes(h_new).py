"""
smooth_bl_nodes.py  —  standalone, no external project imports

Move interior BL quad nodes layer by layer using a metric step-size h:

  For each node b at depth d with inner neighbour a (depth d-1):

    n   = unit vector from a toward b          (perpendicular-to-wall direction)
    h   = 1 / sqrt( nᵀ · M(a) · n )           (metric step size at a in direction n)

  Collect h for every node in the layer, then blend:

    h_avg = mean(h values in layer)
    h_min = min (h values in layer)
    h_new = (1 - t) * h_min  +  t * h_avg      (t supplied by user, default 0.5)

  Place every node b at:

    b_new = a  +  h_new * n_b                  (same direction n_b, blended distance)

  This single h_new is shared across all nodes in the layer, so
  intra-layer tangential edges remain parallel to the interface.

Optional actual-length mode (length_t)
---------------------------------------
  Instead of deriving h_new from the metric field, the caller may supply a
  parametric value t in (0, 1).  When given, each layer's h_new is computed
  from the *actual* (Euclidean, metric-free) a→b edge length instead of the
  metric-based h_b:

    len_avg = mean( |b - a| in the pristine input mesh, over the layer )
    h_new   = t * len_avg

  Placement still follows  b_new = a + h_new * n_b , so the move always
  starts from the already-placed inner boundary-layer node a, and wall /
  interface nodes are still never moved.  If t is not supplied, the script
  falls back to the original blend_t (h_min/h_avg) behaviour.

After b is moved its metric is updated by log-Euclidean interpolation
between metrics[a] and metrics[c] at b's new parametric position along
the a→c chain.  Subsequent layers pick up these updated metrics.

Log-Euclidean interpolation for SPD matrices:
    M(t) = expm( (1-t)*logm(M0) + t*logm(M1) )

Fixed nodes (wall bc + BL/farfield interface) are never moved.

Optional last-layer refinement (refine_last_layer)
-----------------------------------------------------
  Boolean, user-controlled (default off).  After the main layer-by-layer
  pass above, the strip of quads between the last moved layer
  (depth max_depth-1) and the fixed interface (depth max_depth) is
  examined:

    h_new = (1 - blend_t) * h_min + blend_t * h_avg     (same recipe as
                                                          above, evaluated
                                                          at the inner node
                                                          of this last strip)
    h0    = mean actual current distance from inner node to the
            (fixed) interface node, over the strip

  If h_new >= h0, nothing changes (the strip is already fine enough).
  If h_new < h0, a brand-new ring of nodes is inserted at distance h_new
  from the current inner nodes (splitting every quad in the strip into
  two), and the process repeats — using the new ring as the inner side
  and the SAME fixed interface as the outer side — until h_new >= h0 for
  the remaining gap.  The interface nodes themselves are never moved;
  only new interior nodes/quads are added between them and the strip.

Usage
-----
    python smooth_bl_nodes.py [mesh.vol] [metric.mtr] [wall_bc] [blend_t] [length_t] [refine_last_layer] [out.vol]

    length_t may be 'none' (or omitted) to keep the default blend_t behaviour.
    refine_last_layer may be true/false/1/0/yes/no (default false).

Defaults
--------
    mesh.vol          = netgen_quad8783.vol
    metric.mtr        = adj_metric.mtr
    wall_bc           = 2
    blend_t           = 0.5          (0 → all nodes placed at h_min; 1 → all at h_avg)
    length_t          = None         (0,1) → h_new = length_t * average actual a→b length
    refine_last_layer = False        (user's call; see above)
    out.vol           = <stem>_smoothed.vol
"""

import os
import sys
import numpy as np
from collections import defaultdict, deque

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Input / output files  —  edit defaults here
# ---------------------------------------------------------------------------
MESH_FILE   = os.path.join(_HERE, 'netgen_quad8783.vol')
METRIC_FILE = os.path.join(_HERE, 'adj_metric.mtr')
WALL_BC     = 2
BLEND_T     = 0.5    # blend parameter: 0 → h_min for all nodes, 1 → h_avg for all nodes
LENGTH_T    = 0.5   # (0,1) → h_new = LENGTH_T * average actual a→b length; None → use BLEND_T instead
REFINE_LAST_LAYER = True  # True → iteratively insert new BL layers near the interface (see refine_last_layer)
OUT_FILE    = os.path.join(_HERE, 'netgen_quad8783_smooth_6.vol')   # None → <mesh_stem>_smoothed.vol
# ---------------------------------------------------------------------------


# ===========================================================================
# .vol I/O
# ===========================================================================

def read_mesh(mesh_file):
    """
    Parse a Netgen .vol file (2-D surface mesh).

    Returns
    -------
    elements      : list[list[int]]   0-based node IDs per element (tri or quad)
    boundary_data : dict              keyed 0..N-1; each value is a parsed
                                      boundary-edge record (node IDs 0-based)
    vertex_coords : list[[x, y]]      2-D coordinates, 0-based
    """
    with open(mesh_file) as f:
        lines = f.readlines()  # all raw lines of the .vol file

    def find_section(kw):
        # returns the line index of the section keyword so we can offset from it
        try:
            return lines.index(kw + '\n')
        except ValueError:
            raise ValueError(f"Section '{kw}' not found in {mesh_file}")

    # --- surface elements ---
    idx  = find_section('surfaceelements')  # line index of the 'surfaceelements' header
    nelm = int(lines[idx + 1].strip())      # total number of surface elements
    elements = []                           # will hold one list of node IDs per element
    for i in range(nelm):
        tok    = lines[idx + 2 + i].split()    # tokens: surfnr bcnr domin domout np n0 n1 ...
        np_elm = int(tok[4])                   # number of nodes for this element (3=tri, 4=quad)
        nodes  = [int(tok[5 + k]) - 1 for k in range(np_elm)]  # convert 1-based -> 0-based
        elements.append(nodes)

    # --- boundary edges ---
    idx   = find_section('edgesegmentsgi2')  # line index of the boundary-edge section header
    nbdry = int(lines[idx + 1].strip())      # number of boundary edge records
    boundary_data = {}                       # dict mapping record index -> list of parsed fields
    for i in range(nbdry):
        tok    = lines[idx + 2 + i].split()  # raw tokens for one boundary-edge record
        parsed = []                          # mixed int/float list for this record
        for v in tok:
            try:
                parsed.append(int(v))
            except ValueError:
                parsed.append(float(v))
        parsed[2] -= 1   # node p1: 1-based -> 0-based
        parsed[3] -= 1   # node p2: 1-based -> 0-based
        boundary_data[i] = parsed

    # --- vertex coordinates ---
    idx   = find_section('points')           # line index of the 'points' section header
    nvert = int(lines[idx + 1].strip())      # total number of nodes in the mesh
    vertex_coords = []                       # list of [x, y] per node (0-based index)
    for i in range(nvert):
        tok = lines[idx + 2 + i].split()    # x  y  z  tokens (z is always 0 for 2-D)
        vertex_coords.append([float(tok[0]), float(tok[1])])

    return elements, boundary_data, vertex_coords


def export_mesh(out_file, elements, boundary_data, vertex_coords):
    """
    Write a hybrid (tri + quad) 2-D mesh to a Netgen .vol file.
    Node indices in elements and boundary_data must be 0-based on entry.
    """
    nelm  = len(elements)       # total element count (tris + quads)
    nvert = len(vertex_coords)  # total node count
    nbdry = len(boundary_data)  # total boundary-edge record count

    with open(out_file, 'w') as f:
        f.write("mesh3d\ndimension\n2\ngeomtype\n0\n")
        f.write("# surfnr    bcnr    domin    domout    np      p1      p2      p3\n")
        f.write("surfaceelements\n")
        f.write(f"{nelm}\n")

        for elm in elements:                          # triangles first (Netgen requirement)
            if len(elm) == 3:
                f.write(f"{2:>8}{1:>8}{0:>8}{0:>8}{3:>8}"
                        f"{elm[0]+1:>8}{elm[1]+1:>8}{elm[2]+1:>8}\n")
        for elm in elements:                          # quads second
            if len(elm) == 4:
                f.write(f"{2:>8}{1:>8}{0:>8}{0:>8}{4:>8}"
                        f"{elm[0]+1:>8}{elm[1]+1:>8}{elm[2]+1:>8}{elm[3]+1:>8}\n")

        f.write("\n# matnr    np    p1    p2    p3    p4\n")
        f.write("volumeelements\n0\n\n")
        f.write("# surfid       0      p1       p2   trinum1  trinum2 "
                "domin/sf1 domout/sf2 ednr1    dist1     ednr2     dist2\n")
        f.write("edgesegmentsgi2\n")
        f.write(f"{nbdry}\n")
        for k in range(nbdry):
            v = boundary_data[k]  # one boundary-edge record
            f.write(f"{v[0]}\t0\t{v[2]+1}\t{v[3]+1}\t"
                    f"-1\t-1\t1\t0\t"
                    f"{v[8]}\t{v[9]:.16e}\t{v[10]}\t{v[11]:.16e}\n")

        f.write("#     X     Y     Z\n")
        f.write("points\n")
        f.write(f"{nvert}\n")
        for xy in vertex_coords:
            f.write(f"{xy[0]:>24.16e}{xy[1]:>29.16e}{0:>29.16e}\n")

        f.write("materials\n1\n1 domain1\n\nendmesh\n")


# ===========================================================================
# BL interface detection
# ===========================================================================

def find_bl_interface_edges(tris, quads, vertex_coords):
    """
    Return ordered interface edges between the tri far-field and quad BL.
    Nodes shared by both regions define the interface; they are sorted by
    polar angle around their centroid and chained into [a, b] edge pairs.
    """
    tri_nodes  = {n for elm in tris  for n in elm}   # set of all nodes used by triangles
    quad_nodes = {n for elm in quads for n in elm}   # set of all nodes used by quads
    iface      = list(tri_nodes & quad_nodes)        # nodes shared by both regions = interface
    if not iface:
        raise ValueError("No interface nodes found between tri and quad regions.")

    pts      = np.array([vertex_coords[n] for n in iface])  # (N,2) coordinate array of interface nodes
    centroid = pts.mean(axis=0)                              # geometric centre of the interface loop
    angles   = np.arctan2(pts[:, 1] - centroid[1],
                          pts[:, 0] - centroid[0])           # polar angle of each node around centroid
    order    = np.argsort(angles)                            # indices that sort nodes by angle
    ordered  = [iface[i] for i in order][::-1]              # CW ordering of interface nodes

    # restart from trailing-edge node (max x, break ties by min y) for consistent seam
    start   = max(range(len(ordered)),
                  key=lambda i: (vertex_coords[ordered[i]][0],
                                 -vertex_coords[ordered[i]][1]))  # index of the TE node in ordered
    ordered = ordered[start:] + ordered[:start]                   # reordered starting at TE

    edges = [[ordered[i], ordered[(i + 1) % len(ordered)]]
             for i in range(len(ordered))]   # list of [a, b] edge pairs forming the closed loop
    return edges


# ===========================================================================
# Perpendicular edge classification
# ===========================================================================

def ekey(a, b):
    # canonical (sorted) edge key so (a,b) and (b,a) map to the same entry
    return (min(a, b), max(a, b))


def build_e2q(quad_quads):
    """edge -> [(quad_idx, local_edge_idx), ...]"""
    e2q = defaultdict(list)  # maps each canonical edge to the quads that own it
    for qi, quad in enumerate(quad_quads):
        for li in range(4):  # li: local edge index 0..3 within quad qi
            e2q[ekey(quad[li], quad[(li + 1) % 4])].append((qi, li))
    return e2q


def classify_quads(quad_quads, seed_tang_edges, vertex_coords):
    """
    Classify each quad's two edge-pair axes as perpendicular or tangential.

    Axis 0 = local edges 0 & 2   (k % 2 == 0)
    Axis 1 = local edges 1 & 3   (k % 2 == 1)

    Returns perp_axis[qi]: 0, 1, or -1 (unknown).
    """
    n   = len(quad_quads)   # total number of quad elements
    pa  = [-1] * n          # perp_axis per quad: 0, 1, or -1 (unresolved)
    e2q = build_e2q(quad_quads)  # edge -> owning (quad_idx, local_edge_idx) pairs

    queue = deque()  # BFS queue seeded with quads that touch a known tangential edge
    for qi, quad in enumerate(quad_quads):
        for li in range(4):
            if ekey(quad[li], quad[(li + 1) % 4]) in seed_tang_edges:
                pa[qi] = 1 - (li % 2)  # edge li is tang -> opposite axis is perp
                queue.append(qi)
                break

    visited = {qi for qi in range(n) if pa[qi] != -1}  # quads already classified
    while queue:
        qi   = queue.popleft()
        quad = quad_quads[qi]
        cp   = pa[qi]          # perp axis index of the current quad
        for li in range(4):
            k = ekey(quad[li], quad[(li + 1) % 4])
            for nqi, nli in e2q[k]:        # nqi: neighbour quad index; nli: its local edge index
                if nqi == qi or nqi in visited:
                    continue
                # shared edge is perp in current quad iff its axis == cp
                # same axis in neighbour is nli%2, so we derive neighbour's perp axis
                pa[nqi] = nli % 2 if (li % 2 == cp) else 1 - (nli % 2)
                visited.add(nqi)
                queue.append(nqi)

    # Geometric fallback for quads not reachable from any seed edge
    unknown   = [qi for qi in range(n) if pa[qi] == -1]  # quads still unclassified
    if unknown:
        tang_vecs = []  # unit direction vectors of all seed (tangential) edges
        for (a, b) in seed_tang_edges:
            v  = np.array(vertex_coords[b]) - np.array(vertex_coords[a])
            nv = np.linalg.norm(v)
            if nv > 1e-12:
                tang_vecs.append(v / nv)
        if tang_vecs:
            arr        = np.array(tang_vecs)          # (N,2) matrix of unit tangent vectors
            _, evecs   = np.linalg.eigh(arr.T @ arr)  # eigenvectors of covariance; largest = dominant tang dir
            ref        = evecs[:, -1]                 # dominant tangential reference direction
        else:
            ref = np.array([1.0, 0.0])  # fallback: assume x is tangential

        for qi in unknown:
            quad = quad_quads[qi]
            vc   = vertex_coords
            # ax0: mean direction of the axis-0 edge pair (edges 0 and 2)
            ax0  = (np.array(vc[quad[1]]) - np.array(vc[quad[0]])) + \
                   (np.array(vc[quad[3]]) - np.array(vc[quad[2]]))
            # ax1: mean direction of the axis-1 edge pair (edges 1 and 3)
            ax1  = (np.array(vc[quad[2]]) - np.array(vc[quad[1]])) + \
                   (np.array(vc[quad[0]]) - np.array(vc[quad[3]]))
            # whichever axis is more parallel to ref is tangential; the other is perp
            pa[qi] = 1 if abs(np.dot(ax0, ref)) >= abs(np.dot(ax1, ref)) else 0

    return pa


# ===========================================================================
# Metric file
# ===========================================================================

def read_metric(path):
    """
    Header line: 'N 3'
    Body: one line [m11  m12  m22] per node (0-based).
    """
    metrics = []  # list of [m11, m12, m22] per node, same 0-based order as vertex_coords
    with open(path) as f:
        next(f)   # skip the 'N 3' header line
        for line in f:
            t = line.split()  # three whitespace-separated floats per node
            if t:
                metrics.append([float(v) for v in t[:3]])
    return metrics


# ===========================================================================
# Log-Euclidean metric interpolation (closed-form for 2x2 SPD matrices)
# ===========================================================================

def _logm2(M):
    """
    Closed-form matrix logarithm of a 2x2 symmetric positive-definite matrix.

    For a 2x2 SPD matrix with eigenvalues l1, l2 (both > 0):
        logm(M) = V @ diag(log(l1), log(l2)) @ V^T

    M is given as [m11, m12, m22] (the three independent entries of the
    symmetric matrix [[m11, m12], [m12, m22]]).

    Returns the result as [r11, r12, r22].
    """
    m11, m12, m22 = M
    # Eigenvalues of [[m11, m12], [m12, m22]] via the quadratic formula
    tr   = m11 + m22                       # trace
    det  = m11 * m22 - m12 * m12          # determinant
    disc = max(0.0, tr * tr / 4.0 - det)  # discriminant, clamped to avoid sqrt of negative
    sq   = np.sqrt(disc)
    l1   = tr / 2.0 + sq                  # larger eigenvalue
    l2   = tr / 2.0 - sq                  # smaller eigenvalue

    # Clamp eigenvalues away from zero for numerical safety
    l1 = max(l1, 1e-300)
    l2 = max(l2, 1e-300)

    ll1 = np.log(l1)   # log of larger eigenvalue
    ll2 = np.log(l2)   # log of smaller eigenvalue

    if abs(l1 - l2) < 1e-14 * (abs(l1) + abs(l2) + 1e-300):
        # Nearly-isotropic case: logm is just diag(ll1, ll2) in any basis
        # Since l1 ≈ l2 the off-diagonal of logm vanishes
        return [ll1, 0.0, ll2]

    # Eigenvector for l1: solve (M - l1*I) v = 0
    # First column of (M - l1*I): [m11-l1, m12]
    # Use whichever row has larger magnitude for numerical stability
    if abs(m11 - l1) >= abs(m12):
        # Row 0 dominates: v proportional to [-m12, m11-l1]  →  normalise
        vx, vy = -m12, m11 - l1
    else:
        # Row 1 dominates: v proportional to [-(m22-l1), m12]
        vx, vy = -(m22 - l1), m12

    nrm = np.hypot(vx, vy)
    if nrm < 1e-14:
        return [ll1, 0.0, ll2]   # degenerate: fall back to diagonal
    vx /= nrm
    vy /= nrm
    # v = [vx, vy] is the unit eigenvector for l1
    # The matrix logarithm: logm = ll1 * v v^T + ll2 * w w^T
    # where w = [-vy, vx] is the orthogonal eigenvector for l2
    r11 = ll1 * vx * vx + ll2 * vy * vy
    r12 = (ll1 - ll2) * vx * vy
    r22 = ll1 * vy * vy + ll2 * vx * vx
    return [r11, r12, r22]


def _expm2(L):
    """
    Closed-form matrix exponential of a 2x2 symmetric matrix.

    L is given as [l11, l12, l22].  Uses the same eigen-decompose-then-exponentiate
    approach as _logm2, but applies exp to the eigenvalues instead of log.

    Returns [r11, r12, r22].
    """
    l11, l12, l22 = L
    tr   = l11 + l22
    det  = l11 * l22 - l12 * l12
    disc = max(0.0, tr * tr / 4.0 - det)
    sq   = np.sqrt(disc)
    mu1  = tr / 2.0 + sq   # larger eigenvalue of L
    mu2  = tr / 2.0 - sq   # smaller eigenvalue of L

    em1 = np.exp(mu1)   # exp of larger eigenvalue
    em2 = np.exp(mu2)   # exp of smaller eigenvalue

    if abs(mu1 - mu2) < 1e-14 * (abs(mu1) + abs(mu2) + 1e-300):
        # Nearly-equal eigenvalues: expm ≈ exp(mu1) * I  (off-diag ≈ 0)
        return [em1, 0.0, em2]

    # Eigenvector for mu1 (same logic as _logm2)
    if abs(l11 - mu1) >= abs(l12):
        vx, vy = -l12, l11 - mu1
    else:
        vx, vy = -(l22 - mu1), l12

    nrm = np.hypot(vx, vy)
    if nrm < 1e-14:
        return [em1, 0.0, em2]
    vx /= nrm
    vy /= nrm

    r11 = em1 * vx * vx + em2 * vy * vy
    r12 = (em1 - em2) * vx * vy
    r22 = em1 * vy * vy + em2 * vx * vx
    return [r11, r12, r22]


def interp_metric_log_euclidean(M0, M1, t):
    """
    Log-Euclidean interpolation between two 2x2 SPD metrics.

    M(t) = expm( (1 - t) * logm(M0)  +  t * logm(M1) )

    Parameters
    ----------
    M0, M1 : [m11, m12, m22]  metrics at the two bracketing nodes
    t       : float in [0, 1]; t=0 → M0, t=1 → M1

    Returns
    -------
    [r11, r12, r22]  interpolated metric at parameter t
    """
    logM0 = _logm2(M0)  # matrix log of M0
    logM1 = _logm2(M1)  # matrix log of M1
    # Weighted sum in the log-space (the Riemannian geodesic for SPD matrices)
    s = [(1.0 - t) * logM0[i] + t * logM1[i] for i in range(3)]
    return _expm2(s)    # map back to SPD space via matrix exponential


# ===========================================================================
# Metric distance
# ===========================================================================

def mdist(pa, pb, M):
    """sqrt( [dx,dy] [[m11,m12],[m12,m22]] [dx,dy]^T )"""
    dx, dy        = pb[0] - pa[0], pb[1] - pa[1]  # component-wise displacement from pa to pb
    m11, m12, m22 = M                              # symmetric 2x2 metric tensor components
    return np.sqrt(abs(m11*dx*dx + 2.0*m12*dx*dy + m22*dy*dy))


# ===========================================================================
# Depth assignment (BFS from wall)
# ===========================================================================

def assign_depth(perp_adj, seed_nodes):
    depth = {}       # maps each node to its layer index; 0 = wall
    q     = deque()  # BFS frontier
    for n in seed_nodes:
        depth[n] = 0
        q.append(n)
    while q:
        n = q.popleft()
        for nb in perp_adj[n]:   # nb: neighbour connected by a perpendicular edge
            if nb not in depth:
                depth[nb] = depth[n] + 1  # one layer further from wall than n
                q.append(nb)
    return depth


# ===========================================================================
# Smoothing
# ===========================================================================

def smooth(coords, metrics, perp_edge_set, wall_nodes, iface_nodes, blend_t=0.5, length_t=None):
    """
    Layer-by-layer h-based BL node placement with log-Euclidean metric update.

    For each layer at depth d:

      Pass 1 — compute h for every node b in the layer
        n      = unit vector from inner neighbour a (depth d-1) toward b
        ab_len = actual (Euclidean, metric-free) a→b distance in the
                 pristine input mesh — NOT recomputed from already-moved
                 coordinates, so it stays fixed regardless of layer order
        h_b    = 1 / sqrt( nᵀ · M(a) · n )
                 i.e. the metric step size at a in the direction of the a→b edge

      Layer statistics:
        if length_t is None (default):
          h_avg = mean of all h_b in the layer
          h_min = min  of all h_b in the layer
          h_new = (1 - blend_t) * h_min  +  blend_t * h_avg
        else (actual-length mode):
          len_avg = mean of all ab_len in the layer
          h_new   = length_t * len_avg

      Pass 2 — move every node b:
        b_new = a  +  h_new * n_b
        (same direction n_b per node, but the shared scalar h_new)

        This single h_new keeps all tangential (intra-layer) edges
        parallel to the interface after the move.  The move always starts
        from a, the already-placed inner boundary-layer node; wall and
        interface nodes are excluded from nodes_at_d and are never moved.

      Pass 3 — update metrics[b] by log-Euclidean interpolation
        Find b's parametric position t by projecting b's NEW position onto
        the ORIGINAL (pristine) a→c segment — i.e. using coords[a]/coords[c],
        not new_coords[a]/new_coords[c] — then interpolate using the
        ORIGINAL (pristine) metrics at a and c, not their possibly
        already-updated new_metrics values:
        metrics[b] = expm( (1-t)*logm(metrics[a]) + t*logm(metrics[c]) )
        The result is still written into new_metrics[b], so the next layer's
        Pass 1 picks it up as the metric at its inner neighbour.

    Parameters
    ----------
    coords        : list[[x, y]]       original node coordinates
    metrics       : list[[m11,m12,m22]] metric tensors per node
    perp_edge_set : set of (int,int)   canonical perpendicular edges
    wall_nodes    : set of int         fixed wall-boundary nodes
    iface_nodes   : set of int         fixed BL/farfield interface nodes
    blend_t       : float in [0,1]     0 → place all at h_min, 1 → place all at h_avg
                                        (ignored when length_t is not None)
    length_t      : float in (0,1) or None
                                        when set, h_new = length_t * average actual
                                        (metric-free) a→b length for the layer,
                                        overriding the blend_t scheme

    Returns
    -------
    new_coords  : updated coordinate list
    new_metrics : updated metric list (interior nodes interpolated)
    stats       : dict  d -> {'nodes', 'moved', 'h_min', 'h_avg', 'h_new', 'disp'}
    """
    perp_adj = defaultdict(list)  # perpendicular-edge adjacency list
    for (a, b) in perp_edge_set:
        perp_adj[a].append(b)
        perp_adj[b].append(a)

    fixed     = wall_nodes | iface_nodes
    depth     = assign_depth(perp_adj, wall_nodes)  # layer index: 0 = wall
    max_depth = max((d for n, d in depth.items() if n not in wall_nodes), default=0)
    # nodes at max_depth are the interface; they are fixed

    mode_desc = f"length_t={length_t:.4f} (actual-length mode)" if length_t is not None \
                else f"blend_t={blend_t:.4f}"
    print(f"  BL layers: 0 .. {max_depth}  |  fixed nodes: {len(fixed)}"
          f"  (wall={len(wall_nodes)}, iface={len(iface_nodes)})"
          f"  |  {mode_desc}")

    new_coords  = [list(c) for c in coords]   # mutable copy; updated in place
    new_metrics = [list(m) for m in metrics]  # mutable copy; interior values replaced after moves

    stats = {}

    for d in range(1, max_depth):  # layer 0 (wall) and max_depth (iface) are fixed
        nodes_at_d = [n for n, dp in depth.items() if dp == d and n not in fixed]
        if not nodes_at_d:
            continue

        # ------------------------------------------------------------------
        # Pass 1: compute h_b for every moveable node b in this layer
        # ------------------------------------------------------------------
        # node_data[b] = (a, c, nx, ny, h_b, ab_len)
        #   a      : inner neighbour (depth d-1)
        #   c      : outer neighbour (depth d+1); needed later for metric update
        #   nx,ny  : unit vector from a toward b
        #   h_b    : 1 / sqrt( nᵀ M(a) n )
        #   ab_len : actual (Euclidean, metric-free) distance from a to b in the
        #            pristine input mesh (not the already-smoothed coordinates)
        node_data = {}

        for b in nodes_at_d:
            pb = new_coords[b]

            inner = [nb for nb in perp_adj[b] if depth.get(nb) == d - 1]
            outer = [nb for nb in perp_adj[b] if depth.get(nb) == d + 1]

            if not inner or not outer:
                continue  # boundary of perp-edge graph; skip this node

            # Use the first inner/outer neighbour when there are multiple.
            # (Multiple neighbours at the same depth would indicate a non-strip
            # topology; taking the first is consistent and rarely occurs.)
            a = inner[0]
            c = outer[0]

            pa_ = new_coords[a]

            # Unit vector n from a → b (direction follows the current, possibly
            # already-smoothed geometry, same as the metric-based h_b below)
            dx, dy = pb[0] - pa_[0], pb[1] - pa_[1]
            cur_len = np.hypot(dx, dy)
            if cur_len < 1e-14:
                continue  # a and b are coincident; cannot define direction
            nx, ny = dx / cur_len, dy / cur_len

            # Original (pristine, pre-smoothing) a→b length — always taken from
            # the untouched input mesh `coords`, never from `new_coords`, so it
            # is unaffected by any movement already applied earlier in this run.
            oa, ob = coords[a], coords[b]
            ab_len = np.hypot(ob[0] - oa[0], ob[1] - oa[1])

            # Metric step size: h = 1 / sqrt( nᵀ M(a) n )
            m11, m12, m22 = new_metrics[a]
            nMn = m11 * nx * nx + 2.0 * m12 * nx * ny + m22 * ny * ny
            if nMn < 1e-28:
                continue  # degenerate metric in this direction; skip
            h_b = 1.0 / np.sqrt(nMn)

            node_data[b] = (a, c, nx, ny, h_b, ab_len)

        if not node_data:
            stats[d] = {'nodes': len(nodes_at_d), 'moved': 0}
            continue

        # ------------------------------------------------------------------
        # Layer statistics and step size
        # ------------------------------------------------------------------
        h_vals  = [nd[4] for nd in node_data.values()]   # h_b for all valid nodes
        h_avg   = sum(h_vals) / len(h_vals)              # mean  of h values in this layer
        h_min   = min(h_vals)                             # minimum h in this layer
        len_avg = None
        if length_t is not None:
            # Actual-length mode: h_new = t * average actual (metric-free) a→b length
            len_vals = [nd[5] for nd in node_data.values()]
            len_avg  = sum(len_vals) / len(len_vals)
            h_new    = length_t * len_avg
        else:
            h_new = (1.0 - blend_t) * h_min + blend_t * h_avg
        # h_new is the single step size applied to every node in the layer

        # ------------------------------------------------------------------
        # Pass 2: move each node to  a + h_new * n_b
        # ------------------------------------------------------------------
        for b, (a, c, nx, ny, h_b, ab_len) in node_data.items():
            pa_ = new_coords[a]
            new_coords[b][0] = pa_[0] + h_new * nx  # place b at distance h_new from a
            new_coords[b][1] = pa_[1] + h_new * ny  # along the original a→b unit vector

        # ------------------------------------------------------------------
        # Pass 3: update metrics[b] via log-Euclidean interpolation along a→c
        # ------------------------------------------------------------------
        for b, (a, c, nx, ny, h_b, ab_len) in node_data.items():
            pa_old = coords[a]      # a's ORIGINAL (pristine) position, not new_coords
            pc_old = coords[c]      # c's ORIGINAL (pristine) position, not new_coords
            pb_new = new_coords[b]  # b's freshly computed new position

            # Parametric position t of b's new location, projected onto the
            # ORIGINAL a→c segment (pristine coords, not the possibly
            # already-moved new_coords)
            ac     = [pc_old[0] - pa_old[0], pc_old[1] - pa_old[1]]
            ab_new = [pb_new[0] - pa_old[0], pb_new[1] - pa_old[1]]
            ac2    = ac[0] * ac[0] + ac[1] * ac[1]

            if ac2 < 1e-28:
                continue  # a and c coincide; leave metric unchanged

            t_pos = (ab_new[0] * ac[0] + ab_new[1] * ac[1]) / ac2  # projection parameter
            t_pos = max(0.0, min(1.0, t_pos))                        # clamp to [0, 1]

            # Log-Euclidean interpolation using the ORIGINAL (pristine) metrics
            # at a and c, not their possibly already-updated new_metrics values:
            # M(b) = expm((1-t)*logm(metrics[a]) + t*logm(metrics[c]))
            new_metrics[b] = interp_metric_log_euclidean(
                metrics[a], metrics[c], t_pos
            )
            # Subsequent layers that use b as their inner neighbour will read this
            # updated metric, propagating the interpolated field through the BL.

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        # Representative displacement: distance from original b to new b, averaged
        disp_vals = []
        for b, (a, c, nx, ny, h_b, ab_len) in node_data.items():
            ob = coords[b]   # original position before any smoothing
            nb = new_coords[b]
            disp_vals.append(np.hypot(nb[0] - ob[0], nb[1] - ob[1]))
        disp = sum(disp_vals) / len(disp_vals) if disp_vals else 0.0

        stats[d] = {
            'nodes'  : len(nodes_at_d),
            'moved'  : len(node_data),
            'h_min'  : h_min,
            'h_avg'  : h_avg,
            'len_avg': len_avg,
            'h_new'  : h_new,
            'disp'   : disp,
        }
        if length_t is not None:
            print(f"  Layer {d:>3}: {len(node_data):>4} nodes  |  "
                  f"len_avg={len_avg:.4e}  h_new={h_new:.4e}  "
                  f"|disp_avg|={disp:.4e}")
        else:
            print(f"  Layer {d:>3}: {len(node_data):>4} nodes  |  "
                  f"h_min={h_min:.4e}  h_avg={h_avg:.4e}  h_new={h_new:.4e}  "
                  f"|disp_avg|={disp:.4e}")

    return new_coords, new_metrics, stats



"""

def smooth(coords, metrics, perp_edge_set, wall_nodes, iface_nodes):
    Layer-by-layer metric-weighted node smoothing with log-Euclidean
    metric update at each moved node.

    For each layer (depth d from wall):
      1. Collect the metric-weighted displacement from every perp chain
         that passes through that layer.
      2. Average all contributions into a single (avg_dx, avg_dy).
      3. Apply that same vector to EVERY node in the layer.
      4. For each moved node b, compute its parametric position t along
         the edge a -> c and update metrics[b] by log-Euclidean
         interpolation between metrics[a] and metrics[c] at that t.
         This updated metric is available to nodes processed in
         subsequent layers.

    This keeps every tangential (intra-layer) edge parallel to the
    interface after the move.

    Returns new_coords (updated), new_metrics (updated), stats dict.
    perp_adj = defaultdict(list)  # adjacency list restricted to perpendicular edges only
    for (a, b) in perp_edge_set:
        perp_adj[a].append(b)
        perp_adj[b].append(a)

    fixed     = wall_nodes | iface_nodes  # nodes that must never be moved
    depth     = assign_depth(perp_adj, wall_nodes)  # layer index per node (0 = wall)
    max_depth = max((d for n, d in depth.items() if n not in wall_nodes), default=0)
    # max_depth is the interface layer index; nodes there are fixed too

    print(f"  BL layers: 0 .. {max_depth}  |  fixed nodes: {len(fixed)}"
          f"  (wall={len(wall_nodes)}, iface={len(iface_nodes)})")

    new_coords  = [list(c) for c in coords]          # mutable copy of coordinates
    new_metrics = [list(m) for m in metrics]          # mutable copy of metrics; updated after moves

    stats = {}  # per-layer diagnostic info

    for d in range(1, max_depth):  # d: current layer depth (1 = first layer above wall)
        nodes_at_d = [n for n, dp in depth.items() if dp == d and n not in fixed]
        # nodes_at_d: all moveable nodes that sit exactly at depth d
        if not nodes_at_d:
            continue

        all_dx = []  # displacement contributions in x from every chain in this layer
        all_dy = []  # displacement contributions in y from every chain in this layer

        # Also collect, for each node b, the (a, c) pair(s) used so we can
        # interpolate the metric at b's new position afterwards.
        # chain_pairs[b] accumulates (a, c) tuples that contributed to b's move.
        chain_pairs = defaultdict(list)

        for b in nodes_at_d:
            pb    = new_coords[b]  # current coordinates of node b being evaluated

            inner = [nb for nb in perp_adj[b] if depth.get(nb) == d - 1]
            # inner: perp neighbours one layer closer to the wall (depth d-1)
            outer = [nb for nb in perp_adj[b] if depth.get(nb) == d + 1]
            # outer: perp neighbours one layer further from the wall (depth d+1)

            if not inner or not outer:
                continue  # b is at the edge of the perp-edge graph; skip

            for a in inner:
                for c in outer:
                    pa_ = new_coords[a]   # coordinates of wall-side neighbour a
                    pc_ = new_coords[c]   # coordinates of interface-side neighbour c
                    l1  = mdist(pa_, pb, new_metrics[a])  # metric distance a -> b using M at a
                    l2  = mdist(pb, pc_, new_metrics[c])  # metric distance b -> c using M at c
                    tot = l1 + l2
                    if tot < 1e-14:
                        continue
                    w1, w2 = l1 / tot, l2 / tot
                    all_dx.append(w1 * pa_[0] + w2 * pc_[0] - pb[0])
                    all_dy.append(w1 * pa_[1] + w2 * pc_[1] - pb[1])
                    chain_pairs[b].append((a, c))  # record which (a, c) pair was used for b

        if not all_dx:
            stats[d] = {'nodes': len(nodes_at_d), 'moved': 0}
            continue

        avg_dx = sum(all_dx) / len(all_dx)  # single x-shift shared by every node in layer d
        avg_dy = sum(all_dy) / len(all_dy)  # single y-shift shared by every node in layer d

        # --- Apply the shift and update each node's metric ---
        for b in nodes_at_d:
            new_coords[b][0] += avg_dx
            new_coords[b][1] += avg_dy

            # Recompute b's metric by log-Euclidean interpolation.
            # If multiple (a, c) chains contributed to b, average the interpolated
            # log-matrices before exponentiating (still log-Euclidean, just a
            # weighted mean with equal weights across chains).
            pairs = chain_pairs.get(b)
            if not pairs:
                continue  # no chain data for b; leave its metric unchanged

            pb_new = new_coords[b]  # b's updated position

            log_sum = [0.0, 0.0, 0.0]   # accumulator for the log-space metric sum
            n_valid = 0                  # count of valid (a, c) pairs

            for (a, c) in pairs:
                pa_ = new_coords[a]
                pc_ = new_coords[c]

                # Parametric position of b's NEW location along the straight line a -> c.
                # t = 0 at a, t = 1 at c, computed as the projection onto the a-c vector.
                ac  = [pc_[0] - pa_[0], pc_[1] - pa_[1]]  # vector from a to c
                ab  = [pb_new[0] - pa_[0], pb_new[1] - pa_[1]]  # vector from a to b_new
                ac2 = ac[0] * ac[0] + ac[1] * ac[1]       # squared length of a-c

                if ac2 < 1e-28:
                    continue   # a and c coincide; skip this pair

                t = (ab[0] * ac[0] + ab[1] * ac[1]) / ac2  # projection parameter
                t = max(0.0, min(1.0, t))                    # clamp to [0, 1]

                # Log-Euclidean interpolation of the metric at t along a -> c
                M_interp = interp_metric_log_euclidean(new_metrics[a], new_metrics[c], t)

                # Accumulate in log-space for averaging across chains
                log_M = _logm2(M_interp)
                log_sum[0] += log_M[0]
                log_sum[1] += log_M[1]
                log_sum[2] += log_M[2]
                n_valid += 1

            if n_valid > 0:
                # Average the log-matrices and exponentiate back to SPD space
                log_avg = [v / n_valid for v in log_sum]
                new_metrics[b] = _expm2(log_avg)
                # new_metrics[b] is now the log-Euclidean mean of the interpolated metrics
                # from all chains through b; subsequent layers will use this value.

        disp = np.hypot(avg_dx, avg_dy)  # Euclidean magnitude of the layer shift vector
        stats[d] = {'nodes': len(nodes_at_d), 'moved': len(nodes_at_d), 'disp': disp}
        print(f"  Layer {d:>3}: {len(nodes_at_d):>4} nodes  |  "
              f"avg_dx={avg_dx:+.4e}  avg_dy={avg_dy:+.4e}  |disp|={disp:.4e}")

    return new_coords, new_metrics, stats




"""

# ===========================================================================
# Optional last-layer refinement (insert extra BL layers near the interface)
# ===========================================================================

def _canonical_last_layer_quad(quad, depth, max_depth):
    """
    Try all 4 cyclic rotations of `quad` and return the one whose depth
    pattern is [max_depth-1, max_depth, max_depth, max_depth-1], i.e.
    [inner0, outer0, outer1, inner1].  Cyclic rotation preserves winding.
    Returns None if no rotation matches (not a last-layer quad).
    """
    for shift in range(4):
        r     = quad[shift:] + quad[:shift]
        dpat  = [depth.get(n) for n in r]
        if dpat == [max_depth - 1, max_depth, max_depth, max_depth - 1]:
            return r
    return None


def refine_last_layer(coords, metrics, elements, perp_edge_set, wall_nodes, blend_t):
    """
    Iteratively insert new BL quad layers between the last moved layer
    (depth max_depth-1) and the fixed interface (depth max_depth).

    Each round:
      h_new = (1 - blend_t)*h_min + blend_t*h_avg   (metric step, evaluated
                                                      at the current inner
                                                      nodes, same recipe as
                                                      smooth())
      h0    = mean actual distance from inner node to the fixed interface

      h_new >= h0  -> stop (strip is already fine enough)
      h_new <  h0  -> insert a new ring of nodes at distance h_new from the
                      current inner nodes, splitting every quad in the
                      strip into two; the new ring becomes the inner side
                      for the next round, the interface stays the outer side

    Mutates `coords`, `metrics`, `elements` in place (new nodes appended,
    affected quads replaced/added).  Interface nodes are never moved.

    Returns the number of new layers inserted.
    """
    perp_adj = defaultdict(list)
    for (a, b) in perp_edge_set:
        perp_adj[a].append(b)
        perp_adj[b].append(a)

    depth     = assign_depth(perp_adj, wall_nodes)
    max_depth = max((d for n, d in depth.items() if n not in wall_nodes), default=0)
    if max_depth < 1:
        print("  Last-layer refine: no BL strip to refine.")
        return 0

    quads = [e for e in elements if len(e) == 4]

    strip = []  # list of {'quad': <list ref>, 'inner0','outer0','outer1','inner1'}
    for q in quads:
        depths = [depth.get(n) for n in q]
        if depths.count(max_depth - 1) == 2 and depths.count(max_depth) == 2:
            rot = _canonical_last_layer_quad(q, depth, max_depth)
            if rot is None:
                continue
            i0, o0, o1, i1 = rot
            strip.append({'quad': q, 'inner0': i0, 'outer0': o0,
                           'outer1': o1, 'inner1': i1})

    if not strip:
        print("  Last-layer refine: no last-layer quads found.")
        return 0

    rounds = 0
    while strip:
        outer_of = {}
        for cell in strip:
            outer_of[cell['inner0']] = cell['outer0']
            outer_of[cell['inner1']] = cell['outer1']

        per_node = {}   # inner node -> (nx, ny, d_cur, h_b)
        h_vals, gap_vals = [], []
        for node, outer in outer_of.items():
            pa_, pb_ = coords[node], coords[outer]
            dx, dy   = pb_[0] - pa_[0], pb_[1] - pa_[1]
            d_cur    = np.hypot(dx, dy)
            if d_cur < 1e-14:
                continue
            nx, ny   = dx / d_cur, dy / d_cur
            m11, m12, m22 = metrics[node]
            nMn = m11 * nx * nx + 2.0 * m12 * nx * ny + m22 * ny * ny
            if nMn < 1e-28:
                continue
            h_b = 1.0 / np.sqrt(nMn)
            per_node[node] = (nx, ny, d_cur, h_b)
            h_vals.append(h_b)
            gap_vals.append(d_cur)

        if not h_vals:
            break

        h_avg = sum(h_vals) / len(h_vals)
        h_min = min(h_vals)
        h_new = (1.0 - blend_t) * h_min + blend_t * h_avg
        h0    = sum(gap_vals) / len(gap_vals)

        print(f"  Last-layer refine round {rounds + 1}: cells={len(strip)}  "
              f"h_min={h_min:.4e}  h_avg={h_avg:.4e}  h_new={h_new:.4e}  h0={h0:.4e}")

        if h_new >= h0:
            print("  -> h_new >= h0; no further insertion needed.")
            break

        # ------------------------------------------------------------------
        # Build candidate new-node positions/metrics (not yet committed) and
        # check the metric distance from each candidate to the fixed
        # interface node, using the candidate's own log-Euclidean
        # interpolated metric.  If the layer this would create is already
        # at/under the metric's natural unit length (distance < 1/sqrt(2)),
        # skip creating it instead of committing a too-thin layer.
        # ------------------------------------------------------------------
        candidates = {}   # node -> (new_x, new_y, new_metric)
        mdist_vals = []
        for node, (nx, ny, d_cur, h_b) in per_node.items():
            outer   = outer_of[node]
            t_local = max(0.0, min(1.0, h_new / d_cur))
            new_x   = coords[node][0] + h_new * nx
            new_y   = coords[node][1] + h_new * ny
            new_m   = interp_metric_log_euclidean(metrics[node], metrics[outer], t_local)
            candidates[node] = (new_x, new_y, new_m)
            mdist_vals.append(mdist([new_x, new_y], coords[outer], new_m))

        mdist_avg = sum(mdist_vals) / len(mdist_vals)
        mdist_thresh = 1.0 / np.sqrt(2.0)
        print(f"  -> candidate-to-interface metric distance: {mdist_avg:.4f}"
              f"  (threshold {mdist_thresh:.4f})")
        if mdist_avg < mdist_thresh:
            print("  -> metric distance to interface below 1/sqrt(2); skipping new layer.")
            break

        new_id_of = {}
        for node, (new_x, new_y, new_m) in candidates.items():
            coords.append([new_x, new_y])
            metrics.append(new_m)
            new_id_of[node] = len(coords) - 1

        next_strip = []
        for cell in strip:
            i0, o0, o1, i1 = cell['inner0'], cell['outer0'], cell['outer1'], cell['inner1']
            if i0 not in new_id_of or i1 not in new_id_of:
                continue  # degenerate node (skipped above); leave this cell unsplit
            ida, idb = new_id_of[i0], new_id_of[i1]
            cell['quad'][:] = [i0, ida, idb, i1]          # mutate in place -> lower quad
            upper = [ida, o0, o1, idb]
            elements.append(upper)                        # new outer quad
            next_strip.append({'quad': upper, 'inner0': ida, 'outer0': o0,
                                'outer1': o1, 'inner1': idb})

        strip  = next_strip
        rounds += 1

    print(f"  Last-layer refine: inserted {rounds} new layer(s).")
    return rounds


# ===========================================================================
# Main
# ===========================================================================

def _parse_bool(s, default):
    s = s.strip().lower()
    if s in ('', 'none'):
        return default
    if s in ('1', 'true', 'yes', 'y', 't'):
        return True
    if s in ('0', 'false', 'no', 'n', 'f'):
        return False
    raise ValueError(f"Cannot parse boolean from '{s}'")


def main():
    mesh_file   = sys.argv[1] if len(sys.argv) > 1 else MESH_FILE
    metric_file = sys.argv[2] if len(sys.argv) > 2 else METRIC_FILE
    wall_bc     = int(sys.argv[3])   if len(sys.argv) > 3 else WALL_BC
    blend_t     = float(sys.argv[4]) if len(sys.argv) > 4 else BLEND_T

    length_t = LENGTH_T
    if len(sys.argv) > 5 and sys.argv[5].strip().lower() not in ('', 'none'):
        length_t = float(sys.argv[5])

    refine_last = _parse_bool(sys.argv[6], REFINE_LAST_LAYER) if len(sys.argv) > 6 else REFINE_LAST_LAYER

    out_file = sys.argv[7] if len(sys.argv) > 7 else \
               (OUT_FILE or os.path.splitext(mesh_file)[0] + '_smoothed.vol')

    if not (0.0 <= blend_t <= 1.0):
        sys.exit(f"blend_t must be in [0, 1]; got {blend_t}")
    if length_t is not None and not (0.0 < length_t < 1.0):
        sys.exit(f"length_t must be in (0, 1); got {length_t}")

    for p in (mesh_file, metric_file):
        if not os.path.isfile(p):
            sys.exit(f"File not found: {p}")

    # --- Read mesh ---
    print(f"Reading mesh   : {mesh_file}")
    elements, bdry, coords = read_mesh(mesh_file)
    tris  = [e for e in elements if len(e) == 3]
    quads = [e for e in elements if len(e) == 4]
    print(f"  {len(coords)} nodes | {len(tris)} tris | {len(quads)} quads | {len(bdry)} bdry edges")
    if not quads:
        sys.exit("No quad elements found.")

    # --- Perpendicular edge classification ---
    wall_edges = {ekey(v[2], v[3]) for v in bdry.values() if v[0] == wall_bc}

    try:
        iface_edges    = find_bl_interface_edges(tris, quads, coords)
        iface_edge_set = {ekey(a, b) for a, b in iface_edges}
    except (ValueError, IndexError):
        iface_edge_set = set()

    seed_tang = wall_edges | iface_edge_set
    perp_axis = classify_quads(quads, seed_tang, coords)

    perp_edge_set = set()
    for qi, quad in enumerate(quads):
        pa = perp_axis[qi]
        if pa == -1:
            continue
        for li in range(4):
            if li % 2 == pa:
                perp_edge_set.add(ekey(quad[li], quad[(li + 1) % 4]))

    print(f"  Wall edges={len(wall_edges)}  Interface edges={len(iface_edge_set)}"
          f"  Perp edges={len(perp_edge_set)}")

    # --- Fixed node sets ---
    wall_nodes  = {n for (a, b) in wall_edges     for n in (a, b)}
    iface_nodes = {n for (a, b) in iface_edge_set for n in (a, b)}

    # --- Read metric ---
    print(f"Reading metric : {metric_file}")
    metrics = read_metric(metric_file)
    if len(metrics) < len(coords):
        sys.exit(f"Metric has {len(metrics)} rows but mesh has {len(coords)} nodes.")

    # --- Smooth ---
    mode_msg = f"length_t={length_t}" if length_t is not None else f"blend_t={blend_t}"
    print(f"Smoothing ...  ({mode_msg})")
    new_coords, new_metrics, stats = smooth(
        coords, metrics, perp_edge_set, wall_nodes, iface_nodes,
        blend_t=blend_t, length_t=length_t,
    )
    print(f"  Total nodes moved: {sum(s['moved'] for s in stats.values())}")

    # --- Optional last-layer refinement ---
    if refine_last:
        print(f"Refining last layer ...  (refine_last_layer={refine_last})")
        refine_last_layer(new_coords, new_metrics, elements, perp_edge_set, wall_nodes, blend_t)

    # --- Write output ---
    print(f"Writing output : {out_file}")
    export_mesh(out_file, elements, bdry, new_coords)
    print("Done.")


if __name__ == '__main__':
    main()
