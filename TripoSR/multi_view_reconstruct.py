"""
multi_view_reconstruct.py
--------------------------
Feed 2-3 photos of the same object from different angles to TripoSR.

Pipeline
--------
1. Preprocess each image (background removal, foreground crop).
2. Run TripoSR on BEST view only (or optionally average scene codes).
3. Extract ONE clean mesh at high marching-cubes resolution.
4. **Clean up** mesh: remove small disconnected components (floating fragments).
5. **Project real photo colors** back onto each mesh vertex using orthographic
   projection for every input view; blend colors weighted by vertex visibility.
6. Export the final colored mesh.

Usage
-----
cd TripoSR
python multi_view_reconstruct.py view1.png view2.png [view3.png] \\
       --output-dir output_fused \\
       [--mc-resolution 384] \\
       [--foreground-ratio 0.92] \\
       [--no-remove-bg] \\
       [--fusion-mode single_best|average] \\
       [--model-save-format glb]

NOTES
-----
- fusion-mode=single_best (default): uses view 0 for geometry, all views for color.
  This is more stable when views are taken from very different angles.
- fusion-mode=average: averages triplane latent codes (only helps when views are
  very similar / small angular difference).
- The first image in the list should always be the FRONT / most representative view.
"""

import argparse
import logging
import os
import time

import numpy as np
import rembg
import torch
import trimesh
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

# ---------------------------------------------------------------------------
# Logging & timer
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class Timer:
    def __init__(self):
        self.items = {}
        self.scale = 1000.0

    def start(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"[START] {name}")

    def end(self, name: str):
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        delta = (time.time() - self.items.pop(name)) * self.scale
        logging.info(f"[DONE ] {name}  ({delta:.1f} ms)")


timer = Timer()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-view coloured 3D reconstruction with TripoSR."
    )
    p.add_argument("images", nargs="+",
                   help="2 or 3 image paths taken from different angles. "
                        "The FIRST image should be the front/best view.")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device. Falls back to CPU if CUDA unavailable.")
    p.add_argument("--pretrained-model-name-or-path",
                   default="stabilityai/TripoSR")
    p.add_argument("--chunk-size", default=8192, type=int,
                   help="Chunk size for TripoSR renderer (lower = less VRAM).")
    p.add_argument("--mc-resolution", default=384, type=int,
                   help="Marching-cubes resolution (higher = more detail, slower). Default: 384.")
    p.add_argument("--no-remove-bg", action="store_true",
                   help="Skip background removal (use if background already clean/plain).")
    p.add_argument("--foreground-ratio", default=0.92, type=float,
                   help="Foreground crop ratio (0-1). Default: 0.92.")
    p.add_argument("--output-dir", default="output_fused/")
    p.add_argument("--model-save-format", default="glb",
                   choices=["obj", "glb"],
                   help="Output mesh format. Default: glb.")
    p.add_argument("--color-blend-mode", default="weighted",
                   choices=["weighted", "mean"],
                   help="How to blend colors from multiple views. "
                        "'weighted' uses normal-dot-view visibility; 'mean' is equal weight.")
    p.add_argument("--fusion-mode", default="single_best",
                   choices=["single_best", "average"],
                   help="'single_best': use only the first (front) image for geometry shape,  "
                        "use all views for color projection (recommended for wide-angle views). "
                        "'average': average triplane latent codes from all views "
                        "(only works well for closely-spaced views).")
    p.add_argument("--cleanup-threshold", default=0.02, type=float,
                   help="Remove mesh components smaller than this fraction of total volume. "
                        "Set to 0 to disable cleanup. Default: 0.02 (removes <2%% junk).")
    p.add_argument("--mc-threshold", default=15.0, type=float,
                   help="Marching cubes SDF density threshold. LOWER = include more thin/fine "
                        "structures (chair legs, frames). HIGHER = only dense solid parts. "
                        "Default: 15.0 (TripoSR default is 25.0 which cuts thin objects). "
                        "Try 10.0-20.0 for furniture with thin legs.")
    p.add_argument("--rembg-model", default="isnet-general-use",
                   help="rembg model for background removal. 'isnet-general-use' works best "
                        "for real photos with complex backgrounds. Other options: 'u2net', "
                        "'silueta'. Default: isnet-general-use.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
def preprocess_image(image_path, rembg_session, no_remove_bg,
                     foreground_ratio, save_dir, index):
    """Load, optionally remove background, and normalise for TripoSR input."""
    if no_remove_bg:
        image = Image.open(image_path).convert("RGB")
    else:
        raw = Image.open(image_path)
        # alpha_matting=True dramatically improves edges for thin/fine structures
        # (chair legs, frames, grid backrests) by using alpha compositing
        try:
            image = remove_background(
                raw, rembg_session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10,
            )
            logging.info(f"  View {index}: background removed with alpha matting.")
        except Exception as e:
            logging.warning(f"  View {index}: alpha matting failed ({e}), falling back to standard removal.")
            image = remove_background(raw, rembg_session)

        image = resize_foreground(image, foreground_ratio)
        arr = np.array(image).astype(np.float32) / 255.0
        # composite onto grey background (what TripoSR expects)
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        image = Image.fromarray((arr * 255.0).astype(np.uint8))

    os.makedirs(save_dir, exist_ok=True)
    image.save(os.path.join(save_dir, f"input_view_{index}.png"))
    return image


# ---------------------------------------------------------------------------
# Scene-code averaging  (core of the multi-view improvement)
# ---------------------------------------------------------------------------
def average_scene_codes(scene_codes_list: list) -> torch.Tensor:
    """
    Stack per-view scene codes along the batch dim and mean-pool them.
    TripoSR scene codes are triplane feature tensors of shape (1, C, H, W)
    (or similar). Averaging them in latent space fuses complementary
    shape/appearance information from all views into one consistent volume.
    """
    # Each element is (1, *feature_shape); stack along a new batch dim and mean
    stacked = torch.stack([sc for sc in scene_codes_list], dim=0)  # (V, 1, ...)
    averaged = stacked.mean(dim=0)                                  # (1, ...)
    return averaged


# ---------------------------------------------------------------------------
# Real-photo color projection
# ---------------------------------------------------------------------------
def _build_rotation_y(angle_rad: float) -> np.ndarray:
    """Rotation matrix around the Y axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def _get_view_rotations(n_views: int) -> list:
    """
    Assume the user took photos at roughly equal angular spacing around the
    object.  Return one rotation matrix per view.
    For 1 view → front only.
    For 2 views → front + back (180°).
    For 3 views → front + 120° + 240°.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_views, endpoint=False)
    return [_build_rotation_y(a) for a in angles]


def project_real_colors_to_vertices(
    mesh: trimesh.Trimesh,
    preprocessed_images: list,
    blend_mode: str = "weighted",
    ortho_radius: float = 0.87,
) -> np.ndarray:
    """
    Project each mesh vertex into every input image via orthographic projection
    and sample the real pixel colour.  Returns an (N_verts, 4) uint8 RGBA array.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The fused mesh (in TripoSR's canonical coordinate space).
    preprocessed_images : list of PIL.Image
        The preprocessed (grey-bg) versions of every input photo.
    blend_mode : str
        'weighted' weights each view by max(0, n·v)  (visibility dot product).
        'mean'     gives each view equal weight.
    ortho_radius : float
        Extent of the object in TripoSR's normalised coordinate space
        (scene radius).  TripoSR uses ±0.87 by default.
    """
    n_views = len(preprocessed_images)
    rotations = _get_view_rotations(n_views)

    verts = mesh.vertices                            # (N, 3)  float64
    # Compute per-vertex normals (needed for weighted blending)
    try:
        normals = mesh.vertex_normals                # (N, 3)
    except Exception:
        normals = None

    N = len(verts)
    accumulated_color = np.zeros((N, 3), dtype=np.float64)
    accumulated_weight = np.zeros(N, dtype=np.float64)

    for view_idx, (img, R) in enumerate(zip(preprocessed_images, rotations)):
        img_rgb = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        H_img, W_img = img_rgb.shape[:2]

        # Rotate vertices into camera space
        verts_cam = verts @ R.T                      # (N, 3)

        x = verts_cam[:, 0]   # horizontal  in camera space
        y = verts_cam[:, 1]   # vertical
        z = verts_cam[:, 2]   # depth  (positive = in front of camera)

        # Only consider vertices that are in front of the camera
        visible_mask = z > 0  # basic back-face cull by depth

        # Orthographic projection to image UV coordinates
        u = (x / ortho_radius + 1.0) / 2.0   # [0, 1]
        v = (1.0 - y / ortho_radius) / 2.0   # [0, 1]  (Y flipped)

        # Clamp to valid image area
        in_frame_mask = ((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1))
        active_mask = visible_mask & in_frame_mask

        # Pixel coordinates (nearest-neighbour)
        px = (u * (W_img - 1)).astype(int).clip(0, W_img - 1)
        py = (v * (H_img - 1)).astype(int).clip(0, H_img - 1)

        # Sample colours
        sampled = img_rgb[py, px]              # (N, 3)

        # Visibility weight
        if blend_mode == "weighted" and normals is not None:
            # Camera looks along -Z in world after rotation by R
            view_dir_world = R @ np.array([0.0, 0.0, -1.0])  # camera→scene
            vis_weight = np.dot(normals, -view_dir_world).clip(0, 1)
        else:
            vis_weight = np.ones(N, dtype=np.float64)

        # Zero-out back-facing / out-of-frame vertices
        w = vis_weight * active_mask.astype(np.float64)

        accumulated_color += sampled * w[:, None]
        accumulated_weight += w

        logging.info(
            f"  View {view_idx+1}: {int(active_mask.sum())} / {N} "
            f"vertices projected successfully."
        )

    # Fallback: for vertices with no valid projection, use NeRF vertex colors
    no_hit = accumulated_weight == 0.0
    if no_hit.any() and mesh.visual is not None and \
            hasattr(mesh.visual, "vertex_colors") and \
            mesh.visual.vertex_colors is not None and \
            len(mesh.visual.vertex_colors) == N:
        fallback_rgb = mesh.visual.vertex_colors[:, :3].astype(np.float64) / 255.0
        accumulated_color[no_hit] = fallback_rgb[no_hit]
        accumulated_weight[no_hit] = 1.0
        logging.info(
            f"  {int(no_hit.sum())} vertices had no valid projection; "
            f"using TripoSR NeRF colours as fallback."
        )
    elif no_hit.any():
        accumulated_color[no_hit] = 0.5   # neutral grey
        accumulated_weight[no_hit] = 1.0

    # Normalize
    blended = accumulated_color / accumulated_weight[:, None]  # (N, 3) in [0,1]
    blended = blended.clip(0.0, 1.0)

    # Convert to uint8 RGBA
    rgba = np.ones((N, 4), dtype=np.uint8) * 255
    rgba[:, :3] = (blended * 255.0).astype(np.uint8)
    return rgba


# ---------------------------------------------------------------------------
# Mesh cleanup: remove small disconnected islands (floating fragments)
# ---------------------------------------------------------------------------
def cleanup_mesh(mesh: trimesh.Trimesh, threshold: float = 0.02) -> trimesh.Trimesh:
    """
    Remove disconnected components smaller than `threshold` fraction of the
    largest component's volume.  This eliminates floating fragment artifacts
    that TripoSR sometimes produces for thin/complex objects.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    threshold : float
        Components with volume < threshold * largest_volume are removed.
        Set to 0 to disable.
    """
    if threshold <= 0:
        return mesh

    # Split into connected components
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        logging.info("  Mesh cleanup: single component, nothing to remove.")
        return mesh

    # Sort by number of faces descending (proxy for volume)
    components = sorted(components, key=lambda m: len(m.faces), reverse=True)
    largest_faces = len(components[0].faces)
    kept = []
    removed_count = 0
    for comp in components:
        ratio = len(comp.faces) / largest_faces
        if ratio >= threshold:
            kept.append(comp)
        else:
            removed_count += 1

    logging.info(
        f"  Mesh cleanup: kept {len(kept)} component(s), "
        f"removed {removed_count} small fragment(s) "
        f"(threshold={threshold:.0%} of largest)."
    )

    if len(kept) == 1:
        return kept[0]

    # Merge kept components back into one mesh
    merged = trimesh.util.concatenate(kept)
    return merged


# ---------------------------------------------------------------------------
# Optional: lightweight seam-smoothing (vertex color only, no geometry change)
# ---------------------------------------------------------------------------
def smooth_vertex_colors(mesh: trimesh.Trimesh, iterations: int = 2) -> np.ndarray:
    """
    Laplacian smooth the vertex colors to remove projection seams.
    Does NOT change geometry.  Returns smoothed (N, 4) uint8 RGBA.
    """
    colors = mesh.visual.vertex_colors.astype(np.float64)  # (N, 4)
    adj = [set() for _ in range(len(mesh.vertices))]
    for f in mesh.faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[f[i]].add(f[j])

    for _ in range(iterations):
        new_colors = colors.copy()
        for vi, neighbours in enumerate(adj):
            if neighbours:
                nb_list = list(neighbours)
                new_colors[vi] = (colors[vi] + colors[nb_list].mean(axis=0)) / 2.0
        colors = new_colors

    return colors.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if len(args.images) < 2:
        logging.warning(
            "Only one image provided — running single-view reconstruction "
            "(colour projection still applies)."
        )
    elif len(args.images) > 3:
        logging.warning("More than 3 images given; using the first 3.")
        args.images = args.images[:3]

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Device setup
    # ------------------------------------------------------------------
    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"
        logging.info("CUDA not available — using CPU (slower).")

    # ------------------------------------------------------------------
    # Load TripoSR
    # ------------------------------------------------------------------
    timer.start("Loading TripoSR model")
    model = TSR.from_pretrained(
        args.pretrained_model_name_or_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)
    timer.end("Loading TripoSR model")

    rembg_session = None if args.no_remove_bg else rembg.new_session(args.rembg_model)
    logging.info(f"  rembg model: {args.rembg_model if not args.no_remove_bg else 'disabled'}")

    # ------------------------------------------------------------------
    # Step 1 — Preprocess images
    # ------------------------------------------------------------------
    logging.info(f"\n{'='*60}")
    logging.info(f"  Step 1: Preprocessing {len(args.images)} input image(s)")
    logging.info(f"{'='*60}")

    preprocessed_images = []
    for idx, img_path in enumerate(args.images):
        timer.start(f"Preprocess view {idx+1}")
        img = preprocess_image(
            img_path, rembg_session,
            args.no_remove_bg, args.foreground_ratio,
            args.output_dir, idx,
        )
        preprocessed_images.append(img)
        timer.end(f"Preprocess view {idx+1}")

    # ------------------------------------------------------------------
    # Step 2 — Run TripoSR on the relevant image(s)
    # ------------------------------------------------------------------
    logging.info(f"\n{'='*60}")
    logging.info(f"  Step 2: TripoSR inference  (fusion-mode={args.fusion_mode})")
    logging.info(f"{'='*60}")

    if args.fusion_mode == "single_best":
        # Only run TripoSR on the first (best/front) image for geometry.
        # Other views are used only for color projection.
        logging.info(
            "  Using FIRST image for geometry (single_best mode). "
            "All views will contribute to color."
        )
        timer.start("TripoSR inference (best view)")
        with torch.no_grad():
            fused_scene_code = model([preprocessed_images[0]], device=device)
        timer.end("TripoSR inference (best view)")
        logging.info(f"  Scene code shape: {fused_scene_code.shape}")
    else:
        # average mode: run on all views and average the triplane codes
        per_view_scene_codes = []
        for idx, img in enumerate(preprocessed_images):
            timer.start(f"TripoSR inference view {idx+1}")
            with torch.no_grad():
                scene_codes = model([img], device=device)
            per_view_scene_codes.append(scene_codes)
            timer.end(f"TripoSR inference view {idx+1}")

        # ------------------------------------------------------------------
        # Step 3 — Average scene codes in latent space → fused code
        # ------------------------------------------------------------------
        logging.info(f"\n{'='*60}")
        logging.info(f"  Step 3: Averaging {len(per_view_scene_codes)} scene codes")
        logging.info(f"{'='*60}")

        timer.start("Latent averaging")
        if len(per_view_scene_codes) == 1:
            fused_scene_code = per_view_scene_codes[0]
            logging.info("  Single view — no averaging needed.")
        else:
            fused_scene_code = average_scene_codes(per_view_scene_codes)
            logging.info(
                f"  Fused scene code shape: {fused_scene_code.shape}  "
                f"(mean of {len(per_view_scene_codes)} views)"
            )
        timer.end("Latent averaging")

    # ------------------------------------------------------------------
    # Step 4 — Extract ONE mesh from the fused scene code
    # ------------------------------------------------------------------
    logging.info(f"\n{'='*60}")
    logging.info(f"  Step 4: Extracting mesh (mc-resolution={args.mc_resolution})")
    logging.info(f"{'='*60}")

    logging.info(
        f"  MC threshold: {args.mc_threshold} "
        f"(lower = more thin structures, higher = only solid parts; default was 25.0)"
    )
    timer.start("Extract mesh (marching cubes)")
    meshes = model.extract_mesh(
        fused_scene_code,
        has_vertex_color=True,      # also get NeRF colors (used as fallback)
        resolution=args.mc_resolution,
        threshold=args.mc_threshold,  # KEY FIX: was hardcoded 25.0, now tunable
    )
    timer.end("Extract mesh (marching cubes)")

    mesh = meshes[0]
    logging.info(
        f"  Raw mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces."
    )

    # Save intermediate mesh (with NeRF colors — for comparison)
    nerf_mesh_path = os.path.join(
        args.output_dir, f"mesh_nerf_colors.{args.model_save_format}"
    )
    mesh.export(nerf_mesh_path)
    logging.info(f"  NeRF-colored mesh saved → {nerf_mesh_path}")

    # ------------------------------------------------------------------
    # Step 4b — Clean up disconnected fragments
    # ------------------------------------------------------------------
    logging.info(f"\n{'='*60}")
    logging.info(f"  Step 4b: Cleaning up disconnected fragments")
    logging.info(f"{'='*60}")

    timer.start("Mesh cleanup")
    mesh = cleanup_mesh(mesh, threshold=args.cleanup_threshold)
    timer.end("Mesh cleanup")
    logging.info(
        f"  Clean mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces."
    )

    # ------------------------------------------------------------------
    # Step 5 — Project real photo colors onto vertices
    # ------------------------------------------------------------------
    logging.info(f"\n{'='*60}")
    logging.info(f"  Step 5: Projecting real photo colors onto mesh vertices")
    logging.info(f"{'='*60}")

    timer.start("Real photo color projection")
    real_rgba = project_real_colors_to_vertices(
        mesh,
        preprocessed_images,
        blend_mode=args.color_blend_mode,
    )
    timer.end("Real photo color projection")

    # Apply real colors to the mesh
    mesh.visual.vertex_colors = real_rgba
    logging.info("  Real photo colors applied to mesh vertices.")

    # ------------------------------------------------------------------
    # Step 6 — Smooth color seams (optional, lightweight)
    # ------------------------------------------------------------------
    logging.info("  Smoothing vertex color seams (2 Laplacian passes)...")
    try:
        smooth_rgba = smooth_vertex_colors(mesh, iterations=2)
        mesh.visual.vertex_colors = smooth_rgba
        logging.info("  Color smoothing done.")
    except Exception as e:
        logging.warning(f"  Color smoothing skipped: {e}")

    # ------------------------------------------------------------------
    # Step 7 — Export final mesh
    # ------------------------------------------------------------------
    out_path = os.path.join(
        args.output_dir, f"fused_mesh.{args.model_save_format}"
    )
    mesh.export(out_path)

    # Also export OBJ for easy viewing in any 3D software
    obj_path = os.path.join(args.output_dir, "fused_mesh.obj")
    if args.model_save_format != "obj":
        mesh.export(obj_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  ✅  Multi-view 3D Reconstruction Complete!")
    print("=" * 65)
    print(f"  Input views      : {len(args.images)}")
    print(f"  MC resolution    : {args.mc_resolution}")
    print(f"  Vertices         : {len(mesh.vertices):,}")
    print(f"  Faces            : {len(mesh.faces):,}")
    print(f"  Output folder    : {args.output_dir}")
    print(f"  Fused mesh       : {out_path}")
    print(f"  Also as OBJ      : {obj_path}")
    print(f"  NeRF mesh (ref)  : {nerf_mesh_path}")
    print("=" * 65 + "\n")
    print("  TIP: Open the fused mesh in Blender / MeshLab / Windows 3D Viewer")
    print("       to inspect the reconstructed geometry and colors.")
    print()


if __name__ == "__main__":
    main()
