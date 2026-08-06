# Hybrid Mesh Adaptation

Tools for building and refining 2-D **hybrid CFD meshes** (triangles + a
structured quad boundary layer) around airfoil geometries, in
[Netgen](https://ngsolve.org/) `.vol`/`.in2d` format. Built around a NACA /
RAE2822-style airfoil workflow, but the mesh tools are geometry-agnostic.

## What's in here

| File | What it does |
|---|---|
| `gen_quadratic_bezier_in2d.py` | Reads raw airfoil surface points (`airfoil_pts.txt`) and generates a Netgen `.in2d` geometry file, fitting the airfoil boundary with quadratic Bezier segments (via tangent-based control points) and adding a circular farfield boundary. |
| `deformation_aflr.ipynb` | The main hybrid-mesh workflow notebook: builds a quad boundary-layer mesh and a triangular outer mesh with the `meshAdapt` package (anisotropic front-based triangulation, AFLR-style point insertion, metric-driven boundary-layer marching, Delaunay), then stitches/joins them into one hybrid mesh and exports it to `.vol`. |
| `smooth_bl_nodes(h_new).py` | Post-processing step: relaxes/smooths the boundary-layer quad nodes layer by layer, blending each layer's metric-based step size (`h_min`/`h_avg`) so the boundary-layer spacing grows smoothly away from the wall. Operates directly on a `.vol` mesh + `.mtr` metric file. |
| `uniform_refine.py` | Standalone uniform (1→4, "red") refinement of a triangular `.vol` mesh. Keeps curved boundary-layer edges on the true geometry by splitting them at their Bezier parametric midpoint (using the `.in2d` file), rather than the straight-line midpoint. Usable as a CLI tool or as a library (`bisect_refine.py` imports its mesh I/O and Bezier helpers). |
| `bisect_refine.py` | Adaptive mesh refinement via newest-vertex bisection, driven by a per-element error indicator (a `cellwise_error_*.csv` file — e.g. from a solution error estimator). Refines only the elements that need it, instead of the whole mesh. |
| `meshAdapt/` | The mesh-adaptation library that `deformation_aflr.ipynb` depends on: `Triangulation`, Delaunay/AFLR triangle insertion, boundary-layer marching (`blMesh`), the anisotropic metric field, and geometric predicates. |
| `airfoil_pts.txt`, `netgen_quad8783.vol`, `adj_metric.mtr` | Sample input data: raw airfoil coordinates, an example quad boundary-layer mesh, and an example anisotropic metric field, so the scripts run out of the box. |

## Typical pipeline

1. **Geometry**: `gen_quadratic_bezier_in2d.py` turns raw airfoil points into a Netgen `.in2d` geometry file.
2. **Meshing**: Netgen (external tool) meshes the `.in2d` geometry into a `.vol` file; `deformation_aflr.ipynb` builds the hybrid quad/triangle mesh (boundary layer + outer triangulation) on top of that.
3. **Smoothing**: `smooth_bl_nodes(h_new).py` relaxes the boundary-layer node spacing for mesh quality.
4. **Refinement**: `uniform_refine.py` (uniform) or `bisect_refine.py` (error-driven, adaptive) refine the resulting mesh.

Each stage reads/writes plain `.vol` (mesh), `.in2d` (geometry), and `.mtr`
(anisotropic metric field) files, so you can run the pieces independently as
long as you have the right input file for that stage.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ (uses `typing` generics and f-strings throughout).

## Running

**Generate an airfoil geometry file:**
```bash
python gen_quadratic_bezier_in2d.py
```
Reads `airfoil_pts.txt`, writes `RAE2822.in2d` next to the script.

**Build the hybrid mesh:**
```bash
jupyter notebook deformation_aflr.ipynb
```
Run it **from this folder** (`hybrid-mesh-adaptation/`) so the local
`meshAdapt/` package is importable — the first cell adds the current working
directory to `sys.path`.

**Smooth boundary-layer nodes:**
```bash
python "smooth_bl_nodes(h_new).py" [mesh.vol] [metric.mtr] [wall_bc] [blend_t] [length_t] [refine_last_layer] [out.vol]
```
With no arguments it runs on the included sample `netgen_quad8783.vol` /
`adj_metric.mtr`.

**Uniformly refine a mesh:**
```bash
python uniform_refine.py INPUT.vol OUTPUT.vol [--in2d GEOM.in2d] [--bl-bc 2] [--levels 1]
```

**Adaptively refine a mesh (bisection, error-driven):**
```python
from bisect_refine import load_vol, load_eta, bisect_mesh, export_mesh

coord, elem, seg_data = load_vol("mesh.vol")
eta = load_eta("cellwise_error.csv")
coord, elem, seg_data = bisect_mesh(coord, elem, eta, theta=0.5, boundary=seg_data)
export_mesh("refined.vol", coord, elem, seg_data)
```

## Notes

- `deformation_aflr.ipynb` is an exploratory research notebook — expect
  hardcoded example filenames for intermediate `.vol`/`.in2d`/`.mtr` files
  inside individual cells; treat it as a worked example of the pipeline
  rather than a polished script.
- The `meshAdapt/` package here is the minimal set of modules the notebook
  actually imports, pulled out of a larger working copy of that project.
