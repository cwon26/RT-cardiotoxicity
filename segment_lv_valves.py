"""
LV valve segmentation for Ansys Fluent CFD.

Input:  TotalSegmentator NIfTI masks
          --lv      heart_ventricle_left.nii.gz
          --la      heart_atrium_left.nii.gz
          --aorta   aorta.nii.gz

Output: watertight surface mesh of the LV, split into three named zones
          aortic_outlet.stl   (smooth elliptical disc, blood outlet)
          mitral_inlet.stl    (smooth elliptical disc, blood inlet)
          wall.stl            (myocardial wall, hole at each valve)
          lv_combined.stl     (the three above merged, watertight)
          lv_combined.ply     (same, vertex colors tag each zone)

Algorithm:
  1. Marching-cubes on LV mask  -> raw LV surface in patient (mm) coords.
  2. For each neighbor (aorta, LA), flag LV-surface vertices whose distance
     to that neighbor mask < THRESH_MM  ->  valve "patch".
  3. Keep the largest connected component of each patch.
  4. PCA on patch points:  plane normal + in-plane axes (u, v).
     Fit an axis-aligned ellipse (a, b) to the projected patch.
  5. Clip the LV mesh by each plane (keep interior side).  Get ordered hole
     boundary loop of the resulting cut.
  6. Snap each boundary vertex to the ellipse perimeter at its own angle
     -> smooth elliptical opening, while the wall itself is only adjusted
     in a thin ring near the cut.
  7. Fan-triangulate the ellipse interior -> the valve disc.
  8. Merge wall + two discs, weld coincident boundary vertices, orient
     outward, verify watertight.

Dependencies:  numpy, scipy, scikit-image, nibabel, trimesh
               (optional) pymeshfix   -- used only if the merged mesh is
                                         not watertight after welding.

Usage:
    python segment_lv_valves.py \
        --lv    heart_ventricle_left.nii.gz \
        --la    heart_atrium_left.nii.gz \
        --aorta aorta.nii.gz \
        --out   ./lv_cfd

Author: Chungwon Lee
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    label,
    map_coordinates,
)
from skimage.measure import marching_cubes


# ----------------------------- parameters --------------------------------- #

DIST_THRESH_MM = 3.0       # LV-surface vertex within this of neighbor -> patch
PATCH_DILATE_VOX = 1       # extra dilation of neighbor mask before sampling
ELLIPSE_INFLATE = 1.05     # 5% margin around the projected patch
VALVE_N_BOUNDARY = 96      # resampled boundary vertex count per valve
MERGE_TOL_MM = 0.05        # vertex-weld tolerance when combining submeshes


# =========================== I/O & basic geometry ========================= #

def load_mask(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    mask = np.asarray(img.dataobj) > 0.5
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    return mask.astype(np.uint8), img.affine.astype(float), spacing


def voxel_to_world(vox_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    h = np.c_[vox_xyz, np.ones(len(vox_xyz))]
    return (affine @ h.T).T[:, :3]


def world_to_voxel(world_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(affine)
    h = np.c_[world_xyz, np.ones(len(world_xyz))]
    return (inv @ h.T).T[:, :3]


def lv_surface_mesh(lv_mask: np.ndarray, affine: np.ndarray) -> trimesh.Trimesh:
    """Marching-cubes surface in patient (mm) coordinates."""
    # marching_cubes returns vertices in (i,j,k) index space.
    verts, faces, _, _ = marching_cubes(lv_mask.astype(float), level=0.5)
    verts_w = voxel_to_world(verts, affine)
    mesh = trimesh.Trimesh(vertices=verts_w, faces=faces, process=True)
    # Light Laplacian smoothing so the marching-cubes staircase doesn't
    # dominate the patch extraction.  Preserves volume fairly well.
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=8)
    return mesh


# ============================ patch extraction ============================ #

def sample_distance_mm(
    mesh: trimesh.Trimesh,
    neighbor_mask: np.ndarray,
    affine: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """Distance in mm from every LV surface vertex to the neighbor mask."""
    mask = neighbor_mask.astype(bool)
    if PATCH_DILATE_VOX > 0:
        mask = binary_dilation(mask, iterations=PATCH_DILATE_VOX)
    # EDT over the complement, with anisotropic spacing -> distance in mm.
    edt_mm = distance_transform_edt(~mask, sampling=spacing)
    vox = world_to_voxel(mesh.vertices, affine)
    # map_coordinates expects (axis, npts); NIfTI axes match mask axes.
    return map_coordinates(edt_mm, vox.T, order=1, mode="nearest")


def largest_cc_on_surface(mesh: trimesh.Trimesh, flags: np.ndarray) -> np.ndarray:
    """Keep the largest connected component of flagged vertices, following
    mesh-edge adjacency."""
    idx = np.where(flags)[0]
    if len(idx) == 0:
        return flags
    keep = set(idx.tolist())
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in mesh.edges_unique:
        if a in keep and b in keep:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))

    seen: set[int] = set()
    best: list[int] = []
    for s in keep:
        if s in seen:
            continue
        stack = [s]
        comp: list[int] = []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            stack.extend(adj[v])
        if len(comp) > len(best):
            best = comp

    out = np.zeros_like(flags)
    out[best] = True
    return out


# ============================ plane & ellipse ============================= #

@dataclass
class ValveFrame:
    name: str
    center: np.ndarray      # plane origin (mm)
    normal: np.ndarray      # unit, points OUTWARD from LV interior
    u: np.ndarray           # in-plane axis 1 (major)
    v: np.ndarray           # in-plane axis 2 (minor)
    a: float                # semi-major (mm)
    b: float                # semi-minor (mm)


def fit_valve_frame(
    name: str,
    patch_points: np.ndarray,
    lv_centroid: np.ndarray,
) -> ValveFrame:
    center = patch_points.mean(axis=0)
    X = patch_points - center
    # PCA.  Normal = direction of smallest variance.
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[2]
    # Orient outward: from LV centroid toward patch centroid.
    if np.dot(normal, center - lv_centroid) < 0:
        normal = -normal
    u = Vt[0]
    # Re-orthogonalize v (numerically safer than trusting Vt[1] after sign flip).
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    u = np.cross(v, normal)
    u /= np.linalg.norm(u)

    proj = X @ np.stack([u, v], axis=1)
    a = ELLIPSE_INFLATE * float(np.max(np.abs(proj[:, 0])))
    b = ELLIPSE_INFLATE * float(np.max(np.abs(proj[:, 1])))
    # Safety: keep order major >= minor.
    if b > a:
        a, b = b, a
        u, v = v, -u
    return ValveFrame(name, center, normal, u, v, a, b)


# =========================== clip & stitch =============================== #

def clip_keep_interior(
    mesh: trimesh.Trimesh, frame: ValveFrame
) -> trimesh.Trimesh:
    """Remove the cap above the valve plane, leaving an open hole."""
    # slice_mesh_plane keeps the half-space where (x - origin) . normal >= 0.
    # We want to discard the outward side, so flip the normal.
    return trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_origin=frame.center,
        plane_normal=-frame.normal,
        cap=False,
    )


def ordered_boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Return every closed boundary loop (vertex indices, in order)."""
    edges = mesh.edges_sorted
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    if len(boundary) == 0:
        return []
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    loops: list[np.ndarray] = []
    visited_edges: set[tuple[int, int]] = set()

    def _edge_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    for start in list(adj):
        # find an untouched edge from this vertex
        found_start = False
        for nxt in adj[start]:
            if _edge_key(start, nxt) not in visited_edges:
                found_start = True
                break
        if not found_start:
            continue
        loop = [start]
        prev, cur = start, nxt
        visited_edges.add(_edge_key(prev, cur))
        loop.append(cur)
        while cur != start:
            nxts = [n for n in adj[cur] if _edge_key(cur, n) not in visited_edges]
            if not nxts:
                break
            # Prefer the neighbor that is not prev.
            nxt = nxts[0] if nxts[0] != prev else (nxts[1] if len(nxts) > 1 else nxts[0])
            visited_edges.add(_edge_key(cur, nxt))
            prev, cur = cur, nxt
            if cur == start:
                break
            loop.append(cur)
        loops.append(np.array(loop, dtype=np.int64))
    return loops


def pick_valve_loop(
    loops: list[np.ndarray],
    mesh: trimesh.Trimesh,
    frame: ValveFrame,
) -> np.ndarray:
    """Select the boundary loop closest to the valve plane centroid."""
    best, best_d = None, np.inf
    for lp in loops:
        c = mesh.vertices[lp].mean(axis=0)
        d = np.linalg.norm(c - frame.center)
        if d < best_d:
            best_d, best = d, lp
    assert best is not None, "No hole found after clipping."
    return best


def snap_loop_to_ellipse(
    mesh: trimesh.Trimesh,
    loop: np.ndarray,
    frame: ValveFrame,
) -> np.ndarray:
    """Project each loop vertex onto the plane, then push it radially onto
    the ellipse perimeter at its own angle.  Returns the new 3D positions
    (which overwrite the mesh vertices in-place)."""
    P = mesh.vertices[loop] - frame.center
    x = P @ frame.u
    y = P @ frame.v
    theta = np.arctan2(y, x)
    # Ellipse radius in direction theta.
    ca, sa = np.cos(theta), np.sin(theta)
    denom = np.sqrt((frame.b * ca) ** 2 + (frame.a * sa) ** 2)
    r = (frame.a * frame.b) / np.where(denom < 1e-9, 1e-9, denom)
    nx, ny = r * ca, r * sa
    new_xyz = (
        frame.center
        + nx[:, None] * frame.u
        + ny[:, None] * frame.v
    )
    mesh.vertices[loop] = new_xyz
    return new_xyz


def build_ellipse_cap(
    boundary_xyz: np.ndarray, frame: ValveFrame
) -> trimesh.Trimesh:
    """Fan-triangulate the elliptical disc from the centroid to each pair of
    consecutive boundary vertices.  The boundary order matches the loop, so
    the caller can weld by coordinate."""
    n = len(boundary_xyz)
    verts = np.vstack([frame.center[None, :], boundary_xyz])
    faces = np.empty((n, 3), dtype=np.int64)
    for i in range(n):
        j = (i + 1) % n
        faces[i] = (0, 1 + i, 1 + j)
    cap = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Orient so normal matches frame.normal (outward).
    face_n = np.cross(
        verts[faces[0, 1]] - verts[0], verts[faces[0, 2]] - verts[0]
    )
    if np.dot(face_n, frame.normal) < 0:
        cap.faces = cap.faces[:, ::-1]
    return cap


# ============================ main pipeline =============================== #

def process(
    lv_path: Path,
    la_path: Path,
    aorta_path: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading masks")
    lv_mask, affine, spacing = load_mask(lv_path)
    la_mask, _, _ = load_mask(la_path)
    ao_mask, _, _ = load_mask(aorta_path)
    assert lv_mask.shape == la_mask.shape == ao_mask.shape, \
        "LV / LA / aorta masks are not on the same grid."

    print("[2/6] Extracting LV surface (marching cubes + Taubin)")
    mesh = lv_surface_mesh(lv_mask, affine)
    lv_centroid = mesh.vertices.mean(axis=0)
    print(f"      vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

    print("[3/6] Flagging valve patches (distance <= "
          f"{DIST_THRESH_MM} mm)")
    d_ao = sample_distance_mm(mesh, ao_mask, affine, spacing)
    d_la = sample_distance_mm(mesh, la_mask, affine, spacing)
    flag_ao = d_ao <= DIST_THRESH_MM
    flag_la = d_la <= DIST_THRESH_MM
    # If a vertex is near both, assign to whichever neighbor is closer.
    both = flag_ao & flag_la
    flag_ao[both] = d_ao[both] < d_la[both]
    flag_la[both] = ~flag_ao[both]
    flag_ao = largest_cc_on_surface(mesh, flag_ao)
    flag_la = largest_cc_on_surface(mesh, flag_la)
    print(f"      aortic patch : {flag_ao.sum()} verts")
    print(f"      mitral patch : {flag_la.sum()} verts")
    if flag_ao.sum() < 10 or flag_la.sum() < 10:
        raise RuntimeError(
            "Valve patches are too small.  Check that LV/LA/aorta masks "
            "overlap and are on the same grid; try raising DIST_THRESH_MM."
        )

    print("[4/6] Fitting valve planes & ellipses (PCA)")
    frame_ao = fit_valve_frame("aortic_outlet",
                               mesh.vertices[flag_ao], lv_centroid)
    frame_la = fit_valve_frame("mitral_inlet",
                               mesh.vertices[flag_la], lv_centroid)
    for fr in (frame_ao, frame_la):
        print(f"      {fr.name}: center={fr.center.round(1)}, "
              f"a={fr.a:.1f} mm, b={fr.b:.1f} mm, "
              f"n={fr.normal.round(2)}")

    print("[5/6] Clipping LV and building ellipse caps")
    wall = mesh.copy()
    caps: list[tuple[ValveFrame, trimesh.Trimesh]] = []
    # Clip with aortic plane, then mitral plane, in sequence; each clip
    # gives one hole loop, which we snap and cap.
    for fr in (frame_ao, frame_la):
        wall = clip_keep_interior(wall, fr)
        loops = ordered_boundary_loops(wall)
        # The loop nearest this valve centroid is the one we just opened.
        loop = pick_valve_loop(loops, wall, fr)
        boundary_xyz = snap_loop_to_ellipse(wall, loop, fr)
        cap = build_ellipse_cap(boundary_xyz, fr)
        caps.append((fr, cap))

    # Recompute surface-normal orientation for the wall (outward).
    wall.fix_normals()

    print("[6/6] Writing STLs")
    ao_cap = caps[0][1]
    la_cap = caps[1][1]
    wall.export(out_dir / "wall.stl")
    ao_cap.export(out_dir / "aortic_outlet.stl")
    la_cap.export(out_dir / "mitral_inlet.stl")

    # Merged watertight mesh.  Welding coincident vertices seals the holes.
    combined = trimesh.util.concatenate([wall, ao_cap, la_cap])
    combined.merge_vertices(merge_tex=None, merge_norm=None,
                            digits_vertex=int(-np.log10(MERGE_TOL_MM)))
    combined.remove_duplicate_faces()
    combined.remove_degenerate_faces()
    combined.fix_normals()

    if not combined.is_watertight:
        print("      not watertight after weld; trying pymeshfix")
        try:
            import pymeshfix
            fixer = pymeshfix.MeshFix(combined.vertices, combined.faces)
            fixer.repair(verbose=False, joincomp=True, remove_smallest_components=False)
            combined = trimesh.Trimesh(fixer.v, fixer.f, process=True)
            combined.fix_normals()
        except ImportError:
            print("      pymeshfix not installed; leaving mesh as-is")
    print(f"      watertight={combined.is_watertight}, "
          f"volume={combined.volume / 1000.0:.1f} mL")

    combined.export(out_dir / "lv_combined.stl")

    # Colored PLY: red = aortic outlet, blue = mitral inlet, gray = wall.
    colors = np.tile([180, 180, 190, 255], (len(combined.vertices), 1))
    # Vertices inside the aortic / mitral ellipses (within a small tol) get
    # their own color.  Use geometric membership so color survives the weld.
    def _inside(xy_frame: ValveFrame, V: np.ndarray) -> np.ndarray:
        rel = V - xy_frame.center
        on_plane = np.abs(rel @ xy_frame.normal) < 0.5  # mm
        x = rel @ xy_frame.u
        y = rel @ xy_frame.v
        inside = (x / xy_frame.a) ** 2 + (y / xy_frame.b) ** 2 <= 1.0 + 1e-3
        return on_plane & inside

    colors[_inside(frame_ao, combined.vertices)] = [220,  50,  50, 255]
    colors[_inside(frame_la, combined.vertices)] = [ 50, 120, 220, 255]
    combined.visual.vertex_colors = colors
    combined.export(out_dir / "lv_combined.ply")

    print("Done.  Import lv_combined.stl into Fluent Meshing; the three")
    print("named-selection surfaces (wall / aortic_outlet / mitral_inlet)")
    print("are the individually-exported STLs in the same folder.")


# =============================== CLI ===================================== #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lv", required=True, type=Path,
                   help="TotalSegmentator heart_ventricle_left NIfTI")
    p.add_argument("--la", required=True, type=Path,
                   help="TotalSegmentator heart_atrium_left NIfTI")
    p.add_argument("--aorta", required=True, type=Path,
                   help="TotalSegmentator aorta NIfTI")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process(args.lv, args.la, args.aorta, args.out)
