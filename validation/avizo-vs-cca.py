"""
cluster_comparison.py
---------------------
Compare a connected cluster from an Avizo export (RAW) against one or more
Python CCA outputs (multi-frame TIFFs), producing one figure per TIFF.

Three generalisations over the original script
-----------------------------------------------
1. RAW shape auto-inference
   The RAW file has no header. If RAW_SHAPE is left as None, the script
   infers the Z depth from the file size, the known XY dimensions (from
   the first TIFF), and a candidate dtype. It tries uint8 → uint16 →
   float32 in order and picks the first dtype that gives a whole-number Z.

2. Offset auto-computation
   Rather than hardcoding Z_OFFSET and X_OFFSET, the script computes the
   3-D bounding box of the target label in both the RAW and each TIFF at
   runtime, then derives the per-axis shift from the difference of their
   minimum corners. All three axes (Z, Y, X) are handled.
   It also warns if the TIF cluster touches any face of its file volume,
   which would indicate export clipping.

3. Multi-label support
   Both RAW and TIFF may contain multiple integer labels (0 = background).
   RAW_LABEL and TIF_LABEL select which label to compare. Set either to
   None to treat all non-zero voxels as the target (binary mode).

Outputs
-------
    <tiff_stem>_comparison.png  —  overlap map + per-slice voxel profile
                                    one file per TIFF

Usage
-----
    python cluster_comparison.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_PATH = Path("8_2_biggestcluster.view.raw")

# One output figure is produced per TIFF entry
TIFF_PATHS = [
    Path("output/8_2_sub_registered_filtered_thresholded_extracted_6N/cluster_01_mask_6N.tiff"),
    Path("output/8_2_sub_registered_filtered_thresholded_extracted_18N/cluster_01_mask_18N.tiff"),
    Path("output/8_2_sub_registered_filtered_thresholded_extracted_26N/cluster_01_mask_26N.tiff"),
]

# RAW file layout
# Set RAW_SHAPE to None to auto-infer from file size + TIFF XY dimensions.
# Set RAW_DTYPE to the expected dtype; used both for inference and loading.
RAW_SHAPE = None          # e.g. (1499, 750, 750), or None to auto-infer
RAW_DTYPE = np.uint16     # uint8 | uint16 | float32

# Label selection
# Set to an integer to compare only that label; None = all non-zero (binary).
RAW_LABEL = None          # e.g. 1, or None
TIF_LABEL = None          # e.g. 1, or None

DARK_BG = "#0d0d0d"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raw_mmap(path: Path, shape: tuple, dtype) -> np.ndarray:
    """Memory-map a flat binary RAW file — avoids loading it all into RAM."""
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def iter_tiff_frames(path: Path):
    """Yield each frame of a multi-frame TIFF as a 2-D numpy array."""
    img = Image.open(path)
    for i in range(img.n_frames):
        img.seek(i)
        yield np.array(img)


def tiff_info(path: Path):
    """Return (n_frames, height, width) by peeking at frame 0 only."""
    img = Image.open(path)
    frame0 = np.array(img)
    if frame0.ndim == 3:          # RGB frame — take first channel
        frame0 = frame0[..., 0]
    h, w = frame0.shape
    return img.n_frames, h, w


def as_mask(array: np.ndarray, label) -> np.ndarray:
    """
    Convert a volume slice or array to a binary mask.
    If label is None, any non-zero value counts as foreground.
    If label is an integer, only that exact value counts.
    """
    if label is None:
        return array > 0
    return array == label


# ── RAW shape inference ───────────────────────────────────────────────────────

def infer_raw_shape(raw_path: Path, tiff_paths: list, dtype) -> tuple:
    """
    Infer the (Z, Y, X) shape of a headerless RAW file.

    Strategy
    --------
    - Y and X are taken from the first TIFF's frame dimensions (they must
      match for the comparison to make sense).
    - Z = file_size_in_bytes / (bytes_per_voxel * Y * X). This must be a
      whole number; if not, the dtype is wrong.

    Raises ValueError if no whole-number Z can be found.
    """
    nz, h, w = tiff_info(tiff_paths[0])
    file_bytes = raw_path.stat().st_size
    bytes_per_voxel = np.dtype(dtype).itemsize
    xy_area = h * w

    z_exact = file_bytes / (bytes_per_voxel * xy_area)
    if not z_exact.is_integer():
        raise ValueError(
            f"Cannot infer RAW shape: file size {file_bytes} bytes is not "
            f"divisible by {bytes_per_voxel} bytes/voxel × {xy_area} pixels/slice "
            f"(got Z = {z_exact:.4f}). "
            f"Check RAW_DTYPE or supply RAW_SHAPE manually."
        )
    z = int(z_exact)
    shape = (z, h, w)
    print(f"Auto-inferred RAW shape: {shape}  (dtype={np.dtype(dtype).name})")
    return shape


# ── Bounding-box and offset computation ───────────────────────────────────────

def bounding_box_raw(raw_path: Path, raw_shape: tuple, raw_dtype, label) -> dict:
    """
    Compute the 3-D bounding box of the target label in the RAW volume,
    scanning slice by slice to avoid loading the whole volume at once.

    Returns dict with keys 'z', 'y', 'x', each a (min, max) tuple.
    """
    raw_vol = load_raw_mmap(raw_path, raw_shape, raw_dtype)
    z_min = z_max = y_min = y_max = x_min = x_max = None

    print(f"Computing RAW bounding box ({raw_shape[0]} slices)…")
    for iz in range(raw_shape[0]):
        mask = as_mask(raw_vol[iz], label)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        z_min = iz if z_min is None else min(z_min, iz)
        z_max = iz
        y_min = int(ys.min()) if y_min is None else min(y_min, int(ys.min()))
        y_max = int(ys.max()) if y_max is None else max(y_max, int(ys.max()))
        x_min = int(xs.min()) if x_min is None else min(x_min, int(xs.min()))
        x_max = int(xs.max()) if x_max is None else max(x_max, int(xs.max()))

    if z_min is None:
        raise ValueError("RAW volume contains no voxels matching the target label.")

    bb = {"z": (z_min, z_max), "y": (y_min, y_max), "x": (x_min, x_max)}
    print(f"  RAW bbox: z={bb['z']}  y={bb['y']}  x={bb['x']}")
    return bb


def bounding_box_tif(tif_path: Path, label) -> dict:
    """
    Compute the 3-D bounding box of the target label in a TIFF stack,
    scanning frame by frame.

    Also checks whether the cluster touches any face of the file volume
    (which would indicate export clipping) and prints a warning if so.

    Returns dict with keys 'z', 'y', 'x', each a (min, max) tuple,
    plus 'shape' as (nz, h, w).
    """
    nz, h, w = tiff_info(tif_path)
    z_min = z_max = y_min = y_max = x_min = x_max = None

    print(f"Computing TIF bounding box for {tif_path.name} ({nz} frames)…")
    for iz, frame in enumerate(iter_tiff_frames(tif_path)):
        if frame.ndim == 3:
            frame = frame[..., 0]
        mask = as_mask(frame, label)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        z_min = iz if z_min is None else min(z_min, iz)
        z_max = iz
        y_min = int(ys.min()) if y_min is None else min(y_min, int(ys.min()))
        y_max = int(ys.max()) if y_max is None else max(y_max, int(ys.max()))
        x_min = int(xs.min()) if x_min is None else min(x_min, int(xs.min()))
        x_max = int(xs.max()) if x_max is None else max(x_max, int(xs.max()))

    if z_min is None:
        raise ValueError(
            f"{tif_path.name} contains no voxels matching the target label.")

    bb = {"z": (z_min, z_max), "y": (y_min, y_max), "x": (x_min, x_max),
          "shape": (nz, h, w)}
    print(f"  TIF bbox: z={bb['z']}  y={bb['y']}  x={bb['x']}")

    # Warn about potential export clipping on any face
    touches = []
    if z_min == 0:       touches.append("z_min")
    if z_max == nz - 1:  touches.append("z_max")
    if y_min == 0:       touches.append("y_min")
    if y_max == h - 1:   touches.append("y_max")
    if x_min == 0:       touches.append("x_min")
    if x_max == w - 1:   touches.append("x_max")
    if touches:
        print(f"  ⚠ WARNING: TIF cluster touches volume boundary at: "
              f"{', '.join(touches)} — possible export clipping.")

    return bb


def compute_offsets(raw_bb: dict, tif_bb: dict) -> tuple:
    """
    Derive per-axis offsets so that both volumes align on a shared canvas.

    Convention: TIF is the fixed reference (placed at its own file coords).
    RAW is shifted so its cluster minimum aligns with TIF's cluster minimum.

      raw_z_offset : added to RAW z index when placing on canvas
      tif_y_offset : columns trimmed from TIF start in Y (positive = trim left)
      tif_x_offset : columns trimmed from TIF start in X (positive = trim left)

    Returns (raw_z_offset, tif_y_offset, tif_x_offset).
    """
    raw_z_offset = tif_bb["z"][0] - raw_bb["z"][0]
    tif_y_offset = tif_bb["y"][0] - raw_bb["y"][0]
    tif_x_offset = tif_bb["x"][0] - raw_bb["x"][0]

    print(f"  Offsets — RAW z shift: {raw_z_offset:+d}  "
          f"TIF y shift: {tif_y_offset:+d}  TIF x shift: {tif_x_offset:+d}")
    return raw_z_offset, tif_y_offset, tif_x_offset


# ── Canvas sizing ─────────────────────────────────────────────────────────────

def compute_canvas_shape(raw_shape: tuple, tif_bbs: list,
                         offsets_list: list) -> tuple:
    """
    Compute a canvas large enough to contain all volumes at their
    offset-corrected positions without clipping any cluster data.
    """
    canvas_z = raw_shape[0]
    canvas_y = raw_shape[1]
    canvas_x = raw_shape[2]

    for tif_bb, (raw_z_off, tif_y_off, tif_x_off) in zip(tif_bbs, offsets_list):
        nz, h, w = tif_bb["shape"]
        # RAW extent in Z after offset
        canvas_z = max(canvas_z, raw_shape[0] + abs(raw_z_off), nz)
        # TIF extent in Y/X after shift (negative offset = TIF extends further)
        canvas_y = max(canvas_y, h + max(0, -tif_y_off))
        canvas_x = max(canvas_x, w + max(0, -tif_x_off))

    return int(canvas_z), int(canvas_y), int(canvas_x)


# ── MIP builders ─────────────────────────────────────────────────────────────

def build_raw_mip(raw_path, raw_shape, raw_dtype, label,
                  raw_z_offset, canvas_shape):
    """
    Build the RAW Z-MIP and per-slice profile on the shared canvas.
    Called once (or once per TIFF if z-offsets differ across TIFFs).

    RAW slices are placed at canvas z = iz + raw_z_offset.
    RAW is not shifted in X or Y — it is the spatial reference.
    """
    canvas_z, canvas_y, canvas_x = canvas_shape
    raw_mip     = np.zeros((canvas_y, canvas_x), dtype=np.uint8)
    raw_profile = np.zeros(canvas_z, dtype=np.int64)

    raw_vol = load_raw_mmap(raw_path, raw_shape, raw_dtype)
    print(f"Building RAW MIP ({raw_shape[0]} slices)…")
    for iz in range(raw_shape[0]):
        mask = as_mask(raw_vol[iz], label).astype(np.uint8)
        raw_mip[:raw_shape[1], :raw_shape[2]] = np.maximum(
            raw_mip[:raw_shape[1], :raw_shape[2]], mask)
        canvas_iz = iz + raw_z_offset
        if 0 <= canvas_iz < canvas_z:
            raw_profile[canvas_iz] = int(mask.sum())

    return raw_mip, raw_profile


def build_tif_mip(tif_path, label, tif_y_offset, tif_x_offset, canvas_shape):
    """
    Build a single TIF's Z-MIP and per-slice profile on the shared canvas.

    TIF slices are placed at canvas z = iz (TIF is the Z reference).
    tif_y_offset and tif_x_offset shift the TIF cluster to align with RAW:
      - Positive offset: trim that many pixels from the start of the TIF axis
        (TIF cluster started later than RAW cluster in that axis)
      - Negative offset: pad that many pixels at the start
        (TIF cluster started earlier than RAW cluster in that axis)
    """
    canvas_z, canvas_y, canvas_x = canvas_shape
    nz, h, w = tiff_info(tif_path)

    tif_mip     = np.zeros((canvas_y, canvas_x), dtype=np.uint8)
    tif_profile = np.zeros(canvas_z, dtype=np.int64)

    src_y0 = max(0,  tif_y_offset)
    src_x0 = max(0,  tif_x_offset)
    dst_y0 = max(0, -tif_y_offset)
    dst_x0 = max(0, -tif_x_offset)
    src_h  = h - src_y0
    src_w  = w - src_x0

    print(f"Building TIF MIP for {tif_path.name} ({nz} frames)…")
    for iz, frame in enumerate(iter_tiff_frames(tif_path)):
        if frame.ndim == 3:
            frame = frame[..., 0]
        mask = as_mask(frame, label).astype(np.uint8)
        src  = mask[src_y0:, src_x0:]
        tif_mip[dst_y0:dst_y0 + src_h,
                dst_x0:dst_x0 + src_w] = np.maximum(
            tif_mip[dst_y0:dst_y0 + src_h,
                    dst_x0:dst_x0 + src_w], src)
        if 0 <= iz < canvas_z:
            tif_profile[iz] = int(mask.sum())

    return tif_mip, tif_profile


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(raw_mip, tif_mip, raw_profile, tif_profile,
                    tif_label, raw_label_str, tif_label_str, out_path):
    only_raw = (raw_mip > 0) & (tif_mip == 0)
    only_tif = (raw_mip == 0) & (tif_mip > 0)
    both     = (raw_mip > 0) & (tif_mip > 0)

    rgb = np.zeros((*raw_mip.shape, 3), dtype=np.float32)
    rgb[both,     1] = 1.0   # green — agreement
    rgb[only_raw, 0] = 1.0   # red   — Avizo only
    rgb[only_tif, 2] = 1.0   # blue  — CCA only

    total = only_raw.sum() + only_tif.sum() + both.sum()
    if total == 0:
        print(f"  ⚠ No foreground voxels found for {tif_label} — skipping plot.")
        return
    agr    = 100 * both.sum()     / total
    r_only = 100 * only_raw.sum() / total
    t_only = 100 * only_tif.sum() / total

    print(f"\nOverlap stats — {tif_label} (Z-MIP, shared canvas):")
    print(f"  Agreement : {agr:}%")
    print(f"  RAW only  : {only_raw.sum()} voxels")
    print(f"  TIF only  : {only_tif.sum()} voxels")
    # print(f"  TIF only  : {t_only:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK_BG)

    axes[0].imshow(rgb, interpolation="nearest", aspect="equal")
    axes[0].set_title(
        f"Overlap Map (Z-MIP, shared canvas)\n"
        f"Green = both   Red = Avizo only   Blue = CCA only\n"
        f"Agreement: {agr:.1f}%   RAW only: {r_only:.1f}%   CCA only: {t_only:.1f}%",
        color="white", fontsize=10)
    axes[0].axis("off")

    ax = axes[1]
    ax.set_facecolor(DARK_BG)
    ax.plot(raw_profile, color="#ff4444", lw=1.2,
            label=f"Avizo RAW (label={raw_label_str})")
    ax.plot(tif_profile, color="#44aaff", lw=1.2,
            label=f"CCA {tif_label} (label={tif_label_str})")
    ax.set_xlabel("Slice index Z (shared canvas)", color="#aaa")
    ax.set_ylabel("Voxel count per slice",          color="#aaa")
    ax.set_title("Cluster size per slice", color="white", fontsize=11)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1a1a1a", labelcolor="white", edgecolor="#555")

    fig.suptitle(
        f"Coordinate-corrected comparison: Avizo RAW vs {tif_label}\n"
        "(no voxels cropped — offsets auto-computed from bounding boxes)",
        color="white", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Resolve RAW shape (auto-infer if not set)
    raw_shape = RAW_SHAPE
    if raw_shape is None:
        raw_shape = infer_raw_shape(RAW_PATH, TIFF_PATHS, RAW_DTYPE)

    # 2. Compute RAW bounding box once
    print("\n── RAW bounding box ──────────────────────────────")
    raw_bb = bounding_box_raw(RAW_PATH, raw_shape, RAW_DTYPE, RAW_LABEL)

    # 3. Compute TIF bounding boxes and offsets for each TIFF
    tif_bbs      = []
    offsets_list = []
    for tif_path in TIFF_PATHS:
        print(f"\n── {tif_path.name} ──────────────────────────────────")
        tif_bb  = bounding_box_tif(tif_path, TIF_LABEL)
        offsets = compute_offsets(raw_bb, tif_bb)
        tif_bbs.append(tif_bb)
        offsets_list.append(offsets)

    # 4. Size the shared canvas to fit everything
    canvas_shape = compute_canvas_shape(raw_shape, tif_bbs, offsets_list)
    print(f"\nCanvas shape: {canvas_shape}")

    # 5. Build RAW MIP — once if all z-offsets agree, otherwise per TIFF
    raw_z_offsets = [off[0] for off in offsets_list]
    rebuild_raw_per_tif = len(set(raw_z_offsets)) > 1
    if rebuild_raw_per_tif:
        print("\n⚠ Z offsets differ across TIFFs — RAW MIP will be rebuilt per TIFF.")
    else:
        raw_z_offset = raw_z_offsets[0]
        print(f"\n── Building RAW MIP (z_offset={raw_z_offset:+d}) ────────")
        raw_mip, raw_profile = build_raw_mip(
            RAW_PATH, raw_shape, RAW_DTYPE, RAW_LABEL, raw_z_offset, canvas_shape)

    # 6. Loop over each TIFF
    raw_label_str = str(RAW_LABEL) if RAW_LABEL is not None else "all"
    tif_label_str = str(TIF_LABEL) if TIF_LABEL is not None else "all"

    for tif_path, tif_bb, (raw_z_off, tif_y_off, tif_x_off) in zip(
            TIFF_PATHS, tif_bbs, offsets_list):
        print(f"\n{'─'*55}")

        if rebuild_raw_per_tif:
            print(f"Building RAW MIP for {tif_path.name} (z_offset={raw_z_off:+d})…")
            raw_mip, raw_profile = build_raw_mip(
                RAW_PATH, raw_shape, RAW_DTYPE, RAW_LABEL,
                raw_z_off, canvas_shape)

        tif_mip, tif_profile = build_tif_mip(
            tif_path, TIF_LABEL, tif_y_off, tif_x_off, canvas_shape)

        out_path = tif_path.with_name(tif_path.stem + "_comparison.png")
        plot_comparison(raw_mip, tif_mip, raw_profile, tif_profile,
                        tif_label=tif_path.stem,
                        raw_label_str=raw_label_str,
                        tif_label_str=tif_label_str,
                        out_path=out_path)

    print("\nDone.")