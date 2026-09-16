"""
Public Python API for RAFTcorr — programmatic inference without the GUI.

    from raft_dic_gui import RAFTcorr

    model = RAFTcorr.from_checkpoint("models/active/RAFTcorr_large_v1.pth")
    uv = model.predict(ref_img, def_img)               # (H, W, 2) float32
    uvs = model.run_sequence([img0, img1, img2, ...])  # list of (H, W, 2)

Confidence filtering
--------------------
Two filters are enabled by default and mask low-confidence pixels to NaN:

- Photometric (threshold=10.0): warps the deformed image by the predicted UV
  and measures per-pixel mean-RGB L1 error vs the reference.  Pixels where
  the warp error exceeds the threshold are rejected.  Catches: bad texture,
  out-of-plane motion, lighting gradients, crack faces.

- Convergence (threshold=0.5 px): runs RAFT with all iterations visible and
  measures the per-pixel std of the last `convergence_last_n` flow estimates.
  High std means RAFT never settled on a stable answer there.  Catches:
  featureless regions, periodic patterns, large-displacement failures.

Disable either filter by passing its threshold as None:
    model = RAFTcorr.auto(photometric_threshold=None)   # photometric off
    model = RAFTcorr.auto(convergence_threshold=None)   # convergence off
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    DEFAULT_CONTEXT_PADDING,
    DEFAULT_ITERATIONS,
    DEFAULT_TILE_OVERLAP,
    MAX_TILE_PIXELS,
)
from .model import (
    describe_checkpoint,
    discover_models,
    estimate_safe_pmax,
    load_model,
)
from .processing import dic_over_roi_with_tiling

# A frame is either an (H, W, 3) uint8 numpy array or a path to an image file.
Frame = Union[str, "os.PathLike[str]", np.ndarray]


def _load_frame(frame: Frame) -> np.ndarray:
    """Return (H, W, 3) uint8 RGB array; numpy arrays are passed through."""
    if isinstance(frame, np.ndarray):
        img = frame
    else:
        from PIL import Image
        img = np.asarray(Image.open(frame).convert("RGB"))

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _photometric_error(
    ref_arr: np.ndarray, def_arr: np.ndarray, uv: np.ndarray
) -> np.ndarray:
    """
    Warp def_arr by uv and return per-pixel mean-RGB L1 error vs ref_arr.

    Returns (H, W) float32.  Pixels where uv is NaN are sampled at their
    original position (unwarped), so outside-ROI areas are evaluated against
    identity — they will show non-zero error and get masked anyway since uv
    is already NaN there.
    """
    H, W = uv.shape[:2]

    u = torch.from_numpy(np.nan_to_num(uv[..., 0], nan=0.0)).float()
    v = torch.from_numpy(np.nan_to_num(uv[..., 1], nan=0.0)).float()

    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    # Normalize sampling coordinates to [-1, 1] for F.grid_sample
    norm_x = (grid_x + u) * 2.0 / max(W - 1, 1) - 1.0
    norm_y = (grid_y + v) * 2.0 / max(H - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    def_t = torch.from_numpy(def_arr).permute(2, 0, 1).float().unsqueeze(0)  # (1, 3, H, W)
    warped = F.grid_sample(
        def_t, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )
    warped_np = warped.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3)

    return np.mean(np.abs(ref_arr.astype(np.float32) - warped_np), axis=-1).astype(
        np.float32
    )


def _apply_filters(
    uv: np.ndarray,
    ref_arr: np.ndarray,
    def_arr: np.ndarray,
    extras: dict,
    photometric_threshold: Optional[float],
    convergence_threshold: Optional[float],
) -> np.ndarray:
    """Apply enabled confidence filters in-place; returns uv for convenience."""
    if photometric_threshold is not None:
        error = _photometric_error(ref_arr, def_arr, uv)
        with np.errstate(invalid="ignore"):
            uv[~(error <= photometric_threshold)] = np.nan

    if convergence_threshold is not None and "convergence" in extras:
        conv_map = extras["convergence"]
        with np.errstate(invalid="ignore"):
            uv[~(conv_map <= convergence_threshold)] = np.nan

    return uv


class RAFTcorr:
    """
    Minimal programmatic interface to RAFTcorr inference.

    Handles model loading, automatic tile-size selection, tiling, weighted
    fusion, optional smoothing, and per-pixel confidence filtering.
    No GUI, no Flask server, no disk I/O beyond reading images.

    Usage
    -----
    model = RAFTcorr.from_checkpoint("models/active/RAFTcorr_large_v1.pth")
    model = RAFTcorr.auto()  # auto-discover from models/active/

    uv = model.predict(ref_img, def_img)         # (H, W, 2) float32, NaN = masked
    uvs = model.run_sequence(frame_paths)        # list of (H, W, 2)

    Confidence thresholds
    ---------------------
    photometric_threshold : float | None
        Max allowed mean-RGB warp error (0–255 scale).  Default 10.0.
        None disables the filter.
    convergence_threshold : float | None
        Max allowed per-pixel flow std across last `convergence_last_n`
        RAFT iterations, in pixels.  Default 0.5.
        None disables the filter (also skips the extra inference work).
    convergence_last_n : int
        How many tail iterations to use for convergence std.  Default 3.
    """

    def __init__(
        self,
        model,
        metadata,
        device: str,
        p_max_pixels: Optional[int] = None,
        context_padding: int = DEFAULT_CONTEXT_PADDING,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
        iters: int = DEFAULT_ITERATIONS,
        use_smooth: bool = True,
        sigma: float = 2.0,
        photometric_threshold: Optional[float] = 10.0,
        convergence_threshold: Optional[float] = 0.5,
        convergence_last_n: int = 3,
    ) -> None:
        self._model = model
        self._metadata = metadata
        self.device = device
        self.context_padding = context_padding
        self.tile_overlap = tile_overlap
        self.iters = iters
        self.use_smooth = use_smooth
        self.sigma = sigma
        self.photometric_threshold = photometric_threshold
        self.convergence_threshold = convergence_threshold
        self.convergence_last_n = convergence_last_n
        # Auto-size tiles from available VRAM if not overridden.
        self.p_max_pixels = (
            p_max_pixels
            if p_max_pixels is not None
            else estimate_safe_pmax(metadata, device)
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Union[str, "os.PathLike[str]"],
        device: Optional[str] = None,
        **kwargs,
    ) -> "RAFTcorr":
        """Load a specific checkpoint by path."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = str(model_path)
        metadata = describe_checkpoint(model_path)
        model = load_model(model_path, device=device, metadata=metadata)
        return cls(model, metadata, device, **kwargs)

    @classmethod
    def auto(
        cls,
        models_dir: Optional[Union[str, "os.PathLike[str]"]] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "RAFTcorr":
        """
        Auto-discover and load the default checkpoint from models/active/.

        The default is the checkpoint named in models/active/DEFAULT (or the
        RAFTCORR_DEFAULT_MODEL env var).  Without either, discover_models()
        falls back to reverse-alphabetical order.
        """
        entries = discover_models(str(models_dir) if models_dir else None)
        if not entries:
            raise FileNotFoundError(
                "No .pth checkpoints found in models/active/. "
                "Provide an explicit path via RAFTcorr.from_checkpoint()."
            )
        return cls.from_checkpoint(entries[0].path, device=device, **kwargs)

    # ------------------------------------------------------------------
    # Single-pair inference
    # ------------------------------------------------------------------

    def predict(
        self,
        ref: Frame,
        deformed: Frame,
        roi_mask: Optional[np.ndarray] = None,
        tile_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """
        Run DIC on a single image pair.

        Parameters
        ----------
        ref:           Reference image — (H, W, 3) uint8 numpy array or file path.
        deformed:      Deformed image — same format as ref.
        roi_mask:      Boolean or uint8 mask (H, W).  None → full image.
        tile_callback: Optional callable(tile_idx, tile_total) for progress.

        Returns
        -------
        (H, W, 2) float32 array.  [..., 0] = U (horizontal displacement),
        [..., 1] = V (vertical displacement).  NaN = outside ROI or masked
        by an enabled confidence filter.
        """
        ref_arr = _load_frame(ref)
        def_arr = _load_frame(deformed)
        H, W = ref_arr.shape[:2]
        mask = _full_mask(H, W) if roi_mask is None else _coerce_mask(roi_mask)

        disp_full, _, extras = dic_over_roi_with_tiling(
            ref_arr,
            def_arr,
            mask,
            self._model,
            self.device,
            context_padding=self.context_padding,
            tile_overlap=self.tile_overlap,
            p_max_pixels=self.p_max_pixels,
            use_smooth=self.use_smooth,
            sigma=self.sigma,
            iters=self.iters,
            tile_callback=tile_callback,
            return_convergence=(self.convergence_threshold is not None),
            convergence_last_n=self.convergence_last_n,
        )

        uv = disp_full.astype(np.float32)
        return _apply_filters(
            uv, ref_arr, def_arr, extras,
            self.photometric_threshold,
            self.convergence_threshold,
        )

    # ------------------------------------------------------------------
    # Sequence inference (accumulative mode)
    # ------------------------------------------------------------------

    def run_sequence(
        self,
        frames: Sequence[Frame],
        roi_mask: Optional[np.ndarray] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[np.ndarray]:
        """
        Run accumulative DIC over a frame sequence, all relative to frames[0].

        Each element of the returned list is the displacement from frames[0] to
        frames[i+1] — i.e., results[0] is frames[0]→frames[1], etc.

        Parameters
        ----------
        frames:    Sequence of images.  Each entry may be:
                     - a file path (str or Path)
                     - a numpy array (H, W, 3) uint8
                     - mixed — autodetected per element.
                   frames[0] is the reference; frames[1:] are deformed states.
        roi_mask:  Boolean or uint8 mask (H, W).  None → full image.
        progress:  Optional callable(completed_frames, total_frames) called
                   after each deformed frame is processed.

        Returns
        -------
        List of (N-1) displacement arrays, each (H, W, 2) float32.
        NaN = outside ROI or masked by an enabled confidence filter.

        # TODO(incremental): Add `mode="incremental"`, `key_frames: list[int]`,
        # and `key_frame_interval: int` parameters.
        #
        # Incremental mode should:
        #   1. Call predict() between consecutive key frames only.
        #   2. Compose the chain back to frame 0 via
        #      raft_dic_gui.incremental.accumulate_displacement().
        #   3. Cache loaded reference frames so each is decoded once.
        #
        # See DICProcessor.run() in raft_dic_gui/controller.py (the `incremental`
        # branch of the processing loop) for the reference implementation,
        # including OOM-safe frame eviction and median-filter post-processing.
        """
        if len(frames) < 2:
            raise ValueError(
                "run_sequence requires at least 2 frames (reference + 1 deformed)."
            )

        ref_arr = _load_frame(frames[0])
        H, W = ref_arr.shape[:2]
        mask = _full_mask(H, W) if roi_mask is None else _coerce_mask(roi_mask)
        want_convergence = self.convergence_threshold is not None

        total = len(frames) - 1
        results: List[np.ndarray] = []
        for i, frame in enumerate(frames[1:]):
            def_arr = _load_frame(frame)
            disp_full, _, extras = dic_over_roi_with_tiling(
                ref_arr,
                def_arr,
                mask,
                self._model,
                self.device,
                context_padding=self.context_padding,
                tile_overlap=self.tile_overlap,
                p_max_pixels=self.p_max_pixels,
                use_smooth=self.use_smooth,
                sigma=self.sigma,
                iters=self.iters,
                return_convergence=want_convergence,
                convergence_last_n=self.convergence_last_n,
            )
            uv = disp_full.astype(np.float32)
            _apply_filters(
                uv, ref_arr, def_arr, extras,
                self.photometric_threshold,
                self.convergence_threshold,
            )
            results.append(uv)
            if progress is not None:
                progress(i + 1, total)

        return results

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        filters = []
        if self.photometric_threshold is not None:
            filters.append(f"phot<{self.photometric_threshold}")
        if self.convergence_threshold is not None:
            filters.append(f"conv<{self.convergence_threshold}px")
        filter_str = ", ".join(filters) if filters else "no filters"
        return (
            f"RAFTcorr(model={self._metadata.label!r}, "
            f"device={self.device!r}, "
            f"p_max_pixels={self.p_max_pixels}, "
            f"{filter_str})"
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _full_mask(H: int, W: int) -> np.ndarray:
    return np.ones((H, W), dtype=np.uint8)


def _coerce_mask(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
