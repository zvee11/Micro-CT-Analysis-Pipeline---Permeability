from __future__ import annotations

import re
from pathlib import Path

import numpy as np

if not hasattr(np, "string_"):
    np.string_ = np.bytes_

from ahds import AmiraFile
from ahds.data_stream import byterle_decoder, hxzip_decode
from ahds.grammar import detect_format, get_header

from .config import Config


DTYPE_MAP: dict[str, type[np.generic]] = {
    "byte": np.uint8, "uchar": np.uint8, "char": np.int8,
    "short": np.int16, "ushort": np.uint16, "int": np.int32,
    "float": np.float32, "double": np.float64,
}


def parse_dtype_from_header(header: str) -> type[np.generic]:
    match = re.search(r"Lattice\s*\{\s*(\w+)\s+Data", header, flags=re.I)
    token = match.group(1).lower() if match else "byte"
    if token not in DTYPE_MAP:
        raise ValueError(f"Unsupported Avizo dtype token: {token!r}")
    return DTYPE_MAP[token]


def parse_codec_from_header(header: str) -> str:
    if re.search(r"HxZip", header, flags=re.I):
        return "hxzip"
    if re.search(r"(?:ByteRLE|HxByteRLE)", header, flags=re.I):
        return "byterle"
    return "raw"


def parse_spacing_from_header(header: str, shape_zyx: tuple[int, int, int]) -> tuple[float, float, float] | None:
    z_len, y_len, x_len = shape_zyx
    matches = re.findall(
        r"BoundingBox\s+"
        r"([\-0-9\.Ee\+]+)\s+([\-0-9\.Ee\+]+)\s+"
        r"([\-0-9\.Ee\+]+)\s+([\-0-9\.Ee\+]+)\s+"
        r"([\-0-9\.Ee\+]+)\s+([\-0-9\.Ee\+]+)",
        header,
    )
    if not matches:
        return None
    x0, x1, y0, y1, z0, z1 = map(float, matches[-1])
    
    def _edge(extent: float, n: int) -> float:
        return extent / (n-1) if n > 1 else extent
    return (_edge(z1 - z0, z_len), _edge(y1 - y0, y_len), _edge(x1 - x0, x_len))


def label_histogram(vol: np.ndarray) -> dict[int, int]:
    """Count labels in a 3D integer volume, slice-by-slice to bound memory."""
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {vol.shape}")
    if not np.issubdtype(vol.dtype, np.integer):
        raise TypeError(f"Expected an integer label volume, got dtype {vol.dtype}")
    if vol.size == 0:
        return {}
    max_label = int(vol.max())
    if max_label < 0:
        raise ValueError("label_histogram expects non-negative label values")
    counts = np.zeros(max_label + 1, dtype=np.int64)
    for z in range(vol.shape[0]):
        counts += np.bincount(vol[z].ravel(), minlength=counts.size)
    return {int(label): int(count) for label, count in enumerate(counts) if count > 0}


def read_avizo(path: Path, parse_spacing: bool = True, memmap_raw: bool = True):
    path = Path(path)
    af = AmiraFile(str(path), load_streams=False)
    header = get_header(str(path), detect_format(str(path)), header_bytes=180_000)

    x_len, y_len, z_len = map(int, af.header.Lattice.length)
    shape_zyx = (z_len, y_len, x_len)
    n_voxels = x_len * y_len * z_len

    dtype = parse_dtype_from_header(header)
    n_bytes = n_voxels * np.dtype(dtype).itemsize
    codec = parse_codec_from_header(header)

    if codec == "raw" and memmap_raw:
        vol = np.memmap(str(path), dtype=dtype, mode="r", offset=af.meta.header_length, shape=shape_zyx)
    else:
        with open(path, "rb") as f:
            f.seek(af.meta.header_length)
            payload = f.read()
        if codec == "hxzip":
            buf = hxzip_decode(payload, n_bytes)
        elif codec == "byterle":
            buf = byterle_decoder(payload, n_bytes)
        else:
            buf = memoryview(payload)[:n_bytes]
        vol = np.frombuffer(buf, dtype=dtype, count=n_voxels).reshape(shape_zyx)

    spacing = parse_spacing_from_header(header, shape_zyx) if parse_spacing else None
    return vol, spacing, {"dtype": dtype, "codec": codec, "shape_zyx": shape_zyx}


def iter_input_files(cfg: Config) -> list[Path]:
    def _scan_order(p: Path) -> tuple:
        # Sort by the two leading integers in the filename (e.g. "10_0_..."),
        # numerically, so 8_x comes before 10_x. Files that do not match this
        # pattern fall back to lexicographic order after the numbered ones.
        m = re.match(r"(\d+)_(\d+)", p.name)
        if m:
            return (0, int(m.group(1)), int(m.group(2)), p.name)
        return (1, 0, 0, p.name)

    files = sorted(cfg.data_dir.glob(cfg.input_glob), key=_scan_order)
    if not files:
        raise FileNotFoundError(f"No files found in {cfg.data_dir} matching {cfg.input_glob}")
    return files
