# check_codec.py
from pipeline.io import read_avizo
from pathlib import Path

# vol, spacing, info = read_avizo(
#     Path("data/9_2_sub_registered_filtered_thresholded_extracted.am"),
#     parse_spacing=True,
#     memmap_raw=False,   # force full load
# )
# print("shape:", vol.shape)
# print("codec:", info["codec"])
# print("dtype:", info["dtype"])
# print("spacing (um):", tuple(s*1e6 for s in spacing) if spacing else None)
# print("RAM used (MB):", vol.nbytes / 1e6)

import numpy as np, cc3d, os
scans = [('8_2','track'),('8_3','track'),('8_4','track'),('9_0','track'),
         ('9_1','track'),('9_2','track'),('9_3','track'),('9_5','cluster')]
for s, p in scans:
    f = rf'output\{s}_sub_registered_filtered_thresholded_extracted\26N\{p}_03_domain_gas_26N.raw'
    if not os.path.exists(f):
        print(f'{s:5} | NO FILE')
        continue
    v = np.fromfile(f, dtype=np.uint8).reshape(523, 750, 750)
    L = cc3d.connected_components((v == 0).astype(np.uint8), connectivity=26)
    top = set(np.unique(L[0])) - {0}
    bot = set(np.unique(L[-1])) - {0}
    span = top & bot
    span_sizes = sorted([int((L == c).sum()) for c in span], reverse=True)[:3]
    print(f'{s:5} | gas {int((v==0).sum()):>9} | comps {int(L.max()):>4} | '
          f'inlet {len(top):>3} outlet {len(bot):>3} | spanning {len(span)} | sizes {span_sizes}')