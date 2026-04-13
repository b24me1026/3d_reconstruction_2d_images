# 3D Reconstruction from 2D Images

A deep learning pipeline to reconstruct high-fidelity 3D objects from 2D photographs using **TripoSR** and a custom multi-view fusion system.

---

## 📌 Overview

This project implements two approaches to 3D object reconstruction:

1. **Multi-View TripoSR Pipeline** — Takes 2–3 photos of an object from different angles and produces a complete, colored 3D mesh (`.glb` / `.obj`)
2. **Pix3D Bulk Reconstruction** — Batch point cloud reconstruction using a CNN-based encoder-decoder trained on the Pix3D dataset

---

## 📁 Project Structure

```
DC_Project/
├── TripoSR/
│   ├── tsr/                        # Core TripoSR model modules
│   │   ├── system.py               # Main TSR model class
│   │   ├── utils.py                # Background removal, preprocessing
│   │   ├── bake_texture.py         # Texture baking utilities
│   │   └── models/                 # NeRF renderer, isosurface, transformers
│   ├── multi_view_reconstruct.py   # Core multi-view fusion logic
│   ├── run_multi_view.py           # ⭐ Main launcher — edit & run this
│   ├── run.py                      # Single-image TripoSR runner
│   ├── run_triposr_sample.py       # Quick sample test
│   ├── requirements.txt            # Dependencies
│   ├── input_images/               # Place your input photos here
│   └── output_fused/               # 3D mesh outputs saved here
├── bulk_reconstruct.py             # Pix3D batch reconstruction script
├── Dc_final.ipynb                  # Main project notebook
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
cd TripoSR
pip install -r requirements.txt
```

### 2. Set your input images

Open `TripoSR/run_multi_view.py` and edit the image paths:

```python
IMAGE_PATHS = [
    r"path\to\your\front_view.png",   # best angle — used for geometry
    r"path\to\your\side_view.png",    # second angle — used for color
]
```

### 3. Run reconstruction

```bash
cd TripoSR
python run_multi_view.py
```

### 4. View output

Results are saved to `TripoSR/output_fused/`:
- `fused_mesh.glb` — open with Windows 3D Viewer or [gltf.report](https://gltf.report)
- `fused_mesh.obj` — open with Blender or any 3D software

---

## ⚙️ Key Settings (in `run_multi_view.py`)

| Parameter | Default | Description |
|---|---|---|
| `MC_RESOLUTION` | `256` | Mesh detail level (don't exceed 256 on low VRAM GPUs) |
| `MC_THRESHOLD` | `15.0` | Lower = more thin structures preserved |
| `FOREGROUND_RATIO` | `0.92` | How much the object fills the frame |
| `COLOR_BLEND_MODE` | `weighted` | How colors from multiple views are blended |
| `REMBG_MODEL` | `isnet-general-use` | Background removal model |

---

## 🛠 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended: 4GB+ VRAM)
- PyTorch, rembg, trimesh, open3d

---

## 📚 References

- [TripoSR by StabilityAI](https://github.com/VAST-AI-Research/TripoSR)
- [Pix3D Dataset](http://pix3d.csail.mit.edu/)
