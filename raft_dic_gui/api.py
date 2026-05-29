"""
Public Python API for RAFTcorr — programmatic inference without the GUI.

    from raft_dic_gui import RAFTcorr

    model = RAFTcorr.from_checkpoint("models/active/RAFTcorr_large_v1.pth")
    uv = model.predict(ref_img, def_img)               # (H, W, 2) float32
    uvs = model.run_sequence([img0, img1, img2, ...])  # list of (H, W, 2)
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

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


class RAFTcorr:
    """
    Minimal programmatic interface to RAFTcorr inference.

    Handles model loading, automatic tile-size selection, tiling, weighted
    fusion, and optional smoothing.  No GUI, no Flask server, no disk I/O
    beyond reading images.

    Usage
    -----
    # Load a specific checkpoint:
    model = RAFTcorr.from_checkpoint("models/active/RAFTcorr_large_v1.pth")

    # Or auto-discover from models/active/:
    model = RAFTcorr.auto()

    # Single pair → (H, W, 2) float32, NaN outside ROI:
    uv = model.predict(ref_img, def_img)

    # Image sequence, all relative to frames[0]:
    uvs = model.run_sequence(frame_paths, progress=print)
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
    ) -> None:
        self._model = model
        self._metadata = metadata
        self.device = device
        self.context_padding = context_padding
        self.tile_overlap = tile_overlap
        self.iters = iters
        self.use_smooth = use_smooth
        self.sigma = sigma
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
            import torch
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
        Auto-discover and load the first checkpoint from models/active/.

        discover_models() returns entries sorted reverse-alphabetically, so a
        naming convention like RAFTcorr_large_v2 takes priority over v1.
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
        [..., 1] = V (vertical displacement).  NaN outside the ROI.
        """
        ref_arr = _load_frame(ref)
        def_arr = _load_frame(deformed)

        H, W = ref_arr.shape[:2]
        mask = _full_mask(H, W) if roi_mask is None else _coerce_mask(roi_mask)

        disp_full, _ = dic_over_roi_with_tiling(
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
        )
        return disp_full.astype(np.float32)

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

        total = len(frames) - 1
        results: List[np.ndarray] = []
        for i, frame in enumerate(frames[1:]):
            def_arr = _load_frame(frame)
            disp_full, _ = dic_over_roi_with_tiling(
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
            )
            results.append(disp_full.astype(np.float32))
            if progress is not None:
                progress(i + 1, total)

        return results

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RAFTcorr(model={self._metadata.label!r}, "
            f"device={self.device!r}, "
            f"p_max_pixels={self.p_max_pixels})"
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _full_mask(H: int, W: int) -> np.ndarray:
    return np.ones((H, W), dtype=np.uint8)


def _coerce_mask(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
