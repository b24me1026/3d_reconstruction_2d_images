"""
run_multi_view.py
-----------------
Helper / example launcher for multi_view_reconstruct.py.

Edit the IMAGE_PATHS list below with the actual paths to your 2-3 object photos,
then just run:

    cd Replica_X
    python run_multi_view.py
"""

import os
import subprocess
import sys

# ============================================================
# ✏️  EDIT THESE:  paths to your 2 or 3 input photos
# NOTE: Put the BEST / most complete view FIRST — it is used
#       for geometry shape in single_best mode.
#       Images were originally uploaded in reverse, fixed here.
# ============================================================
IMAGE_PATHS = [
    r"C:\Users\HP\Desktop\phtoto project\slipper_front.png",   # [0] geometry view (best angle)
    r"C:\Users\HP\Desktop\phtoto project\slipper_top.png",  # [1] used for color projection
    # r"examples\view3.png", # optional third view (uncomment to use)
]

# ============================================================
# ✏️  Optional settings
# ============================================================
OUTPUT_DIR       = "output_fused"
MC_RESOLUTION    = 256         # ⚠️  DO NOT change to 384/512 — causes CUDA OOM on this GPU
FOREGROUND_RATIO = 0.92        # how much of the frame the object should fill
MODEL_FORMAT     = "glb"       # "glb" or "obj"
COLOR_BLEND_MODE = "weighted"  # "weighted" (uses normals) or "mean" (equal)
FUSION_MODE      = "single_best"  # "single_best" = stable; "average" = only for close-angle views
MC_THRESHOLD     = 15.0        # KEY: lower = more thin structures (was 25.0 = cuts chair legs!)
                               # Try 10.0 if legs still missing, 20.0 if too noisy
REMBG_MODEL      = "isnet-general-use"  # better than u2net for real-world cluttered backgrounds
CHUNK_SIZE       = 4096        # ⚠️  DO NOT remove — reduces VRAM usage during mesh extraction

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    # Validate that the image files exist
    for p in IMAGE_PATHS:
        if not os.path.exists(p):
            print(f"\n❌  Image not found: {p}")
            print("    Please edit run_multi_view.py and set the correct paths.\n")
            sys.exit(1)

    cmd = [
        sys.executable,
        "multi_view_reconstruct.py",
        *IMAGE_PATHS,
        "--output-dir",        OUTPUT_DIR,
        "--mc-resolution",     str(MC_RESOLUTION),
        "--foreground-ratio",  str(FOREGROUND_RATIO),
        "--model-save-format", MODEL_FORMAT,
        "--color-blend-mode",  COLOR_BLEND_MODE,
        "--fusion-mode",       FUSION_MODE,
        "--mc-threshold",      str(MC_THRESHOLD),
        "--rembg-model",       REMBG_MODEL,
        "--chunk-size",        str(CHUNK_SIZE),
    ]

    print("\n" + "=" * 65)
    print("  🚀  Launching Replica_X multi-view 3D reconstruction …")
    print("=" * 65)
    print("  Images :", IMAGE_PATHS)
    print("  Output :", OUTPUT_DIR)
    print("  MC res :", MC_RESOLUTION)
    print("=" * 65 + "\n")

    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)
