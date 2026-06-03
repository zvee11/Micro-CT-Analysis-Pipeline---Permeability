# check_codec.py
from pipeline.io import read_avizo
from pathlib import Path

vol, spacing, info = read_avizo(
    Path("data/9_2_sub_registered_filtered_thresholded_extracted.am"),
    parse_spacing=True,
    memmap_raw=False,   # force full load
)
print("shape:", vol.shape)
print("codec:", info["codec"])
print("dtype:", info["dtype"])
print("spacing (um):", tuple(s*1e6 for s in spacing) if spacing else None)
print("RAM used (MB):", vol.nbytes / 1e6)