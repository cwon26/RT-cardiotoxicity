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
  1. Marching-cubes on LV mask -> raw LV surface in patient (mm) coords.
     Light Taubin smoothing is applied so the staircase does not bias
     face-normal sampling, but stays small enough that face centroids
     remain on the LV/non-LV interface.
  2. For each LV face, sample the aorta and LA masks at
       face_centroid + step * outward_face_normal.
     Faces touching aorta -> aortic patch; touching LA -> mitral.
     Everything else -> wall.  Each patch is reduced to its largest
     face-connected component to remove sampling noise.
  3. Carve the wall mesh = (LV mesh) - (aortic faces) - (mitral faces).
     The two holes left behind are the natural valve annuli.
  4. For each annulus loop (a closed ring of mesh edges), fit a plane
     (PCA) and an axis-aligned ellipse (max in-plane extent).  The
     ellipse size is therefore determined by anatomy, not by a manual
     threshold.
  5. Snap each annulus vertex to the ellipse perimeter at its own
     angle -> smooth elliptical opening; the wall is only nudged in
     a thin ring near the annulus.
  6. Fan-triangulate the ellipse interior from its center -> valve disc.
  7. Merge wall + two discs, weld coincident boundary vertices, orient
     outward, verify watertight.

Dependencies:  numpy, scipy, scikit-image, nibabel, trimesh
               (optional) pymeshfix -- only invoked if merging leaves
                                       the mesh non-watertight.

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
from skimage.measure import marching_cubes


# ----------------------------- parameters --------------------------------- #

LABEL_WALL = 0
LABEL_AORTIC = 1
LABEL_MITRAL = 2

# Mask-sampling step along the outward face normal.  ~1 voxel is enough to
# leave the LV mask and land in the neighbor mask if they are touching.
SAMPLE_STEP_MM = 1.5

# Light Taubin smoothing of the raw marching-cubes surface.  Few iterations
# so face centroids stay on the LV/non-LV interface.
TAUBIN_ITERS = 3
TAUBIN_LAMBDA = 0.5
TAUBIN_NU = -0.53

# When merging wall + valve discs, weld vertices closer than this (mm).
MERGE_TOL_MM = 0.05


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
    verts, faces, _, _ = marching_cubes(lv_mask.astype(float), level=0.5)
    verts_w = voxel_to_world(verts, affine)
    mesh = trimesh.Trimesh(vertices=verts_w, faces=faces, process=True)
    trimesh.smoothing.filter_taubin(
        mesh, lamb=TAUBIN_LAMBDA, nu=TAUBIN_NU, iterations=TAUBIN_ITERS
    )
    return mesh


# ====================== mask-intersection face labels ==================== #

def sample_mask_at_world(
    mask: np.ndarray, points_world: np.ndarray, affine: np.ndarray
) -> np.ndarray:
    """Return a bool array: True if the nearest voxel of `mask` is set."""
    iv = np.round(world_to_voxel(points_world, affine)).astype(int)
    sh = mask.shape
    ok = (
        (iv[:, 0] >= 0) & (iv[:, 0] < sh[0]) &
        (iv[:, 1] >= 0) & (iv[:, 1] < sh[1]) &
        (iv[:, 2] >= 0) & (iv[:, 2] < sh[2])
    )
    out = np.zeros(len(points_world), dtype=bool)
    if ok.any():
        v = iv[ok]
        out[ok] = mask[v[:, 0], v[:, 1], v[:, 2]] > 0
    return out


def label_faces_by_neighbor(
    mesh: trimesh.Trimesh,
    aorta_mask: np.ndarray,
    la_mask: np.ndarray,
    affine: np.ndarray,
    step_mm: float = SAMPLE_STEP_MM,
) -> np.ndarray:
    """For every LV face, decide whether its outside is aorta / LA / wall.

    Uses two probe points per face (step and 2*step along the outward
    normal) and accepts the label if EITHER probe lands in the neighbor
    mask.  The OR makes labelling tolerant to small registration
    mismatches between LV and aorta/LA segmentations.
    """
    centroids = mesh.triangles_center
    normals = mesh.face_normals
    p1 = centroids + step_mm * normals
    p2 = centroids + 2.0 * step_mm * normals
    in_a = (
        sample_mask_at_world(aorta_mask, p1, affine)
        | sample_mask_at_world(aorta_mask, p2, affine)
    )
    in_l = (
        sample_mask_at_world(la_mask, p1, affine)
        | sample_mask_at_world(la_mask, p2, affine)
    )

    labels = np.full(len(mesh.faces), LABEL_WALL, dtype=np.int8)
    # If a face is adjacent to both (rare, registration overlap), keep the
    # one with the deeper hit (probe p1 already inside).
    both = in_a & in_l
    in_a_only = in_a & ~in_l
    in_l_only = in_l & ~in_a
    labels[in_a_only] = LABEL_AORTIC
    labels[in_l_only] = LABEL_MITRAL
    if both.any():
        # Re-probe at p1 only; whichever is set there wins.  If both still
        # set, default to aorta (closer to LVOT).
        a1 = sample_mask_at_world(aorta_mask, p1[both], affine)
        l1 = sample_mask_at_world(la_mask, p1[both], affine)
        idx = np.where(both)[0]
        labels[idx[l1 & ~a1]] = LABEL_MITRAL
        labels[idx[a1 | (a1 & l1)]] = LABEL_AORTIC
    return labels


def keep_largest_face_component(
    mesh: trimesh.Trimesh, mask: np.ndarray
) -> np.ndarray:
    """Given a boolean per-face mask, return a new mask containing only
    the largest face-adjacency-connected component."""
    fi = np.where(mask)[0]
    if len(fi) <= 1:
        return mask.copy()
    keep = set(int(x) for x in fi)
    fa = mesh.face_adjacency  # pairs of adjacent face indices
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in fa:
        a_i, b_i = int(a), int(b)
        if a_i in keep and b_i in keep:
            adj[a_i].append(b_i)
            adj[b_i].append(a_i)

    seen: set[int] = set()
    best: list[int] = []
    for s in keep:
        if s in seen:
            continue
        stack, comp = [s], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x])
        if len(comp) > len(best):
            best = comp

    out = np.zeros_like(mask)
    out[best] = True
    return out


# ============================ carve & loop =============================== #

def carve_wall(
    mesh: trimesh.Trimesh, face_labels: np.ndarray
) -> trimesh.Trimesh:
    """Drop labelled (valve) faces, return the clean wall mesh."""
    keep = face_labels == LABEL_WALL
    wall = mesh.submesh([np.where(keep)[0]], append=True)
    if isinstance(wall, list):
        wall = wall[0]
    # Drop the largest connected wall component only (in case stray faces
    # got severed during carving).
    comps = wall.split(only_watertight=False)
    if len(comps) > 1:
        wall = max(comps, key=lambda c: len(c.faces))
    return wall


def boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Return every closed boundary loop (vertex indices, in order)."""
    edges = mesh.edges_sorted
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    bnd = uniq[counts == 1]
    if len(bnd) == 0:
        return []
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in bnd:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    used: set[tuple[int, int]] = set()

    def _ekey(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    loops: list[np.ndarray] = []
    for start in list(adj):
        # find an unused edge from start
        nxt = None
        for n in adj[start]:
            if _ekey(start, n) not in used:
                nxt = n
                break
        if nxt is None:
            continue
        loop = [start]
        used.add(_ekey(start, nxt))
        prev, cur = start, nxt
        loop.append(cur)
        while cur != start:
            cands = [n for n in adj[cur] if _ekey(cur, n) not in used]
            if not cands:
                break
            # avoid backtracking when more than one option exists
            n = cands[0] if cands[0] != prev else (
                cands[1] if len(cands) > 1 else cands[0]
            )
            used.add(_ekey(cur, n))
            prev, cur = cur, n
            if cur == start:
                break
            loop.append(cur)
        if len(loop) >= 4:
            loops.append(np.array(loop, dtype=np.int64))
    return loops


def assign_loops_to_valves(
    wall: trimesh.Trimesh,
    loops: list[np.ndarray],
    aortic_face_centroids: np.ndarray,
    mitral_face_centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match each loop to the valve whose carved face-cluster is closest."""
    if len(loops) < 2:
        raise RuntimeError(
            f"Expected two boundary loops after carving, got {len(loops)}. "
            "Mask labelling probably failed; check SAMPLE_STEP_MM and that "
            "LV / aorta / LA masks share the same NIfTI grid."
        )
    aortic_c = aortic_face_centroids.mean(axis=0)
    mitral_c = mitral_face_centroids.mean(axis=0)
    # rank loops by combined distance, pick the best aortic, then best
    # mitral from the remaining.
    centers = [wall.vertices[lp].mean(axis=0) for lp in loops]
    da = [np.linalg.norm(c - aortic_c) for c in centers]
    dm = [np.linalg.norm(c - mitral_c) for c in centers]
    ai = int(np.argmin(da))
    remaining = [i for i in range(len(loops)) if i != ai]
    mi = remaining[int(np.argmin([dm[i] for i in remaining]))]
    return loops[ai], loops[mi]


# ========================== plane & ellipse fit =========================== #

@dataclass
class ValveFrame:
    name: str
    center: np.ndarray   # plane origin (mm)
    normal: np.ndarray   # unit, points outward from LV interior
    u: np.ndarray        # in-plane major axis
    v: np.ndarray        # in-plane minor axis
    a: float             # semi-major (mm)
    b: float             # semi-minor (mm)


def fit_valve_frame_from_loop(
    name: str, loop_xyz: np.ndarray, lv_centroid: np.ndarray
) -> ValveFrame:
    """Fit plane (PCA) and axis-aligned ellipse to the annulus loop.

    Semi-axes are the maximum signed extent along each principal axis -
    the smallest ellipse that contains the natural annulus, with no
    arbitrary inflation factor.
    """
    center = loop_xyz.mean(axis=0)
    X = loop_xyz - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[2]
    if np.dot(normal, center - lv_centroid) < 0:
        normal = -normal
    u = Vt[0]
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    u = np.cross(v, normal)
    u /= np.linalg.norm(u)

    proj = X @ np.stack([u, v], axis=1)
    a = float(np.max(np.abs(proj[:, 0])))
    b = float(np.max(np.abs(proj[:, 1])))
    if b > a:
        a, b = b, a
        u, v = v, -u
    return ValveFrame(name, center, normal, u, v, a, b)


def snap_loop_to_ellipse(
    mesh: trimesh.Trimesh, loop: np.ndarray, frame: ValveFrame
) -> np.ndarray:
    """Project each loop vertex onto the plane, push it radially to the
    ellipse perimeter at its own angle.  Mutates `mesh.vertices`."""
    P = mesh.vertices[loop] - frame.center
    x = P @ frame.u
    y = P @ frame.v
    theta = np.arctan2(y, x)
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
    """Fan-triangulate the elliptical disc from its center.  Boundary
    vertices stay where the wall placed them so the cap welds cleanly."""
    n = len(boundary_xyz)
    verts = np.vstack([frame.center[None, :], boundary_xyz])
    faces = np.empty((n, 3), dtype=np.int64)
    for i in range(n):
        j = (i + 1) % n
        faces[i] = (0, 1 + i, 1 + j)
    cap = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
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
    lv_mask, affine, _ = load_mask(lv_path)
    la_mask, _, _ = load_mask(la_path)
    ao_mask, _, _ = load_mask(aorta_path)
    if not (lv_mask.shape == la_mask.shape == ao_mask.shape):
        raise ValueError("LV / LA / aorta masks must share the same grid.")

    print("[2/6] Extracting LV surface (marching cubes + Taubin)")
    mesh = lv_surface_mesh(lv_mask, affine)
    lv_centroid = mesh.vertices.mean(axis=0)
    print(f"      vertices={len(mesh.vertices)}  faces={len(mesh.faces)}")

    print("[3/6] Labelling faces by neighbor mask "
          f"(probe step={SAMPLE_STEP_MM} mm)")
    labels = label_faces_by_neighbor(mesh, ao_mask, la_mask, affine)
    # Each valve patch -> largest connected face component (drops noise).
    a_mask = keep_largest_face_component(mesh, labels == LABEL_AORTIC)
    m_mask = keep_largest_face_component(mesh, labels == LABEL_MITRAL)
    labels[(labels == LABEL_AORTIC) & ~a_mask] = LABEL_WALL
    labels[(labels == LABEL_MITRAL) & ~m_mask] = LABEL_WALL
    n_a, n_m = int(a_mask.sum()), int(m_mask.sum())
    print(f"      aortic faces : {n_a}")
    print(f"      mitral faces : {n_m}")
    if n_a < 8 or n_m < 8:
        raise RuntimeError(
            "Valve face patches are too small.  Verify that LV touches "
            "both aorta and LA in your TotalSegmentator masks; raising "
            "SAMPLE_STEP_MM (currently "
            f"{SAMPLE_STEP_MM}) may help if there is a small gap."
        )
    aortic_centroids = mesh.triangles_center[a_mask]
    mitral_centroids = mesh.triangles_center[m_mask]

    print("[4/6] Carving wall and extracting annulus loops")
    wall = carve_wall(mesh, labels)
    loops = boundary_loops(wall)
    print(f"      hole loops found: {len(loops)} "
          f"(sizes={[len(lp) for lp in loops]})")
    aortic_loop, mitral_loop = assign_loops_to_valves(
        wall, loops, aortic_centroids, mitral_centroids
    )

    print("[5/6] Fitting ellipses to annulus loops")
    frame_ao = fit_valve_frame_from_loop(
        "aortic_outlet", wall.vertices[aortic_loop], lv_centroid
    )
    frame_la = fit_valve_frame_from_loop(
        "mitral_inlet", wall.vertices[mitral_loop], lv_centroid
    )
    for fr, lp in ((frame_ao, aortic_loop), (frame_la, mitral_loop)):
        print(f"      {fr.name}: a={fr.a:.1f} mm  b={fr.b:.1f} mm  "
              f"area={np.pi * fr.a * fr.b:.0f} mm^2  "
              f"loop_n={len(lp)}")

    # Snap each annulus to its own ellipse perimeter, then cap.
    ao_xyz = snap_loop_to_ellipse(wall, aortic_loop, frame_ao)
    la_xyz = snap_loop_to_ellipse(wall, mitral_loop, frame_la)
    ao_cap = build_ellipse_cap(ao_xyz, frame_ao)
    la_cap = build_ellipse_cap(la_xyz, frame_la)
    wall.fix_normals()

    print("[6/6] Writing STLs")
    wall.export(out_dir / "wall.stl")
    ao_cap.export(out_dir / "aortic_outlet.stl")
    la_cap.export(out_dir / "mitral_inlet.stl")

    combined = trimesh.util.concatenate([wall, ao_cap, la_cap])
    combined.merge_vertices(
        digits_vertex=int(-np.log10(MERGE_TOL_MM))
    )
    combined.remove_duplicate_faces()
    combined.remove_degenerate_faces()
    combined.fix_normals()

    if not combined.is_watertight:
        print("      not watertight after weld; trying pymeshfix")
        try:
            import pymeshfix
            fixer = pymeshfix.MeshFix(combined.vertices, combined.faces)
            fixer.repair(verbose=False, joincomp=True,
                         remove_smallest_components=False)
            combined = trimesh.Trimesh(fixer.v, fixer.f, process=True)
            combined.fix_normals()
        except ImportError:
            print("      pymeshfix not installed; leaving mesh as-is")
    print(f"      watertight={combined.is_watertight}  "
          f"volume={combined.volume / 1000.0:.1f} mL")

    combined.export(out_dir / "lv_combined.stl")

    # Tagged PLY: red = aortic outlet, blue = mitral inlet, gray = wall.
    colors = np.tile([180, 180, 190, 255], (len(combined.vertices), 1))
    def _on_ellipse(fr: ValveFrame, V: np.ndarray) -> np.ndarray:
        rel = V - fr.center
        on_plane = np.abs(rel @ fr.normal) < 0.5  # mm
        x = rel @ fr.u
        y = rel @ fr.v
        inside = (x / fr.a) ** 2 + (y / fr.b) ** 2 <= 1.0 + 1e-3
        return on_plane & inside

    colors[_on_ellipse(frame_ao, combined.vertices)] = [220,  50,  50, 255]
    colors[_on_ellipse(frame_la, combined.vertices)] = [ 50, 120, 220, 255]
    combined.visual.vertex_colors = colors
    combined.export(out_dir / "lv_combined.ply")

    print("Done.  Import lv_combined.stl into Fluent Meshing; the three")
    print("named-selection surfaces (wall / aortic_outlet / mitral_inlet)")
    print("are the individually-exported STLs in the same folder.")


# =============================== CLI ===================================== #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
