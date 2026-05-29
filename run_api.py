"""
Minimal example: use RAFTcorr as a Python module (no GUI).

Usage
-----
    python run_api.py

Expects at least two images in IMAGES_DIR.  Edit the paths below to match
your data.  The model is auto-selected from models/active/; override with
from_checkpoint() if you need a specific one.
"""

import numpy as np
from raft_dic_gui import RAFTcorr

# --- Configuration --------------------------------------------------------

IMAGES_DIR = "path/to/your/images"   # folder containing a numbered image sequence
MODEL_PATH = "models/active/RAFTcorr_large_v1.pth"  # or use RAFTcorr.auto()

# --------------------------------------------------------------------------


def main():
    # Load model (auto-picks device: CUDA if available, else CPU).
    # To use a specific checkpoint:   model = RAFTcorr.from_checkpoint(MODEL_PATH)
    # To auto-discover:               model = RAFTcorr.auto()
    model = RAFTcorr.from_checkpoint(MODEL_PATH)
    print(model)

    # --- Single pair ---------------------------------------------------------

    import os
    img_files = sorted(
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"))
    )

    if len(img_files) < 2:
        raise RuntimeError(f"Need at least 2 images in {IMAGES_DIR!r}")

    uv = model.predict(img_files[0], img_files[1])
    print(f"Single pair displacement: {uv.shape}  dtype={uv.dtype}")
    print(f"  U  range: {np.nanmin(uv[..., 0]):.3f} to {np.nanmax(uv[..., 0]):.3f} px")
    print(f"  V  range: {np.nanmin(uv[..., 1]):.3f} to {np.nanmax(uv[..., 1]):.3f} px")

    # --- Optional: supply a numpy array directly (skips file I/O) ------------

    # from PIL import Image
    # ref = np.asarray(Image.open(img_files[0]).convert("RGB"))
    # def_ = np.asarray(Image.open(img_files[1]).convert("RGB"))
    # uv = model.predict(ref, def_)

    # --- Optional: restrict to an ROI mask -----------------------------------

    # H, W = uv.shape[:2]
    # roi = np.zeros((H, W), dtype=np.uint8)
    # roi[100:H-100, 100:W-100] = 1        # central crop
    # uv = model.predict(img_files[0], img_files[1], roi_mask=roi)

    # --- Sequence (all frames relative to frame 0) ---------------------------

    def log_progress(done, total):
        print(f"  frame {done}/{total}")

    print(f"\nRunning sequence over {len(img_files)} frames...")
    results = model.run_sequence(img_files, progress=log_progress)

    print(f"Sequence complete: {len(results)} displacement fields")
    for i, uv_i in enumerate(results):
        u_mean = np.nanmean(uv_i[..., 0])
        v_mean = np.nanmean(uv_i[..., 1])
        print(f"  frame {i+1:03d}: mean U={u_mean:+.3f} px  mean V={v_mean:+.3f} px")

    # Results are plain numpy (H, W, 2) float32 — do whatever you like:
    # np.savez("displacements.npz", *results)


if __name__ == "__main__":
    main()
