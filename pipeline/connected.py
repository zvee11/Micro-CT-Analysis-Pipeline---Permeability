from __future__ import annotations

import gc
import logging
from typing import Callable

import cc3d
import numpy as np

from .config import CC3D_CONNECTIVITY, CONNECTIVITY_NAME


class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        parent = self.parent
        while parent.get(x, x) != x:
            px = parent[x]
            parent[x] = parent[px]
            x = parent[x]
        return x if x in parent else x

    def union(self, a: int, b: int) -> None:
        if a == b:
            return
        self.add(a)
        self.add(b)

        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def seam_equivalences(prev_plane: np.ndarray | None, curr_plane: np.ndarray) -> set[tuple[int, int]]:
    if prev_plane is None:
        return set()

    h, w = prev_plane.shape
    pairs = set()

    for dy in (-1, 0, 1):
        y0 = slice(max(0, dy), min(h, h + dy))
        y1 = slice(max(0, -dy), min(h, h - dy))
        for dx in (-1, 0, 1):
            x0 = slice(max(0, dx), min(w, w + dx))
            x1 = slice(max(0, -dx), min(w, w - dx))
            a = prev_plane[y0, x0]
            b = curr_plane[y1, x1]
            m = (a > 0) & (b > 0)
            if m.any():
                pairs.update(zip(a[m].tolist(), b[m].tolist()))
    return pairs


def topn_gas_cc(
    vol: np.ndarray,
    gas_label: int,
    connectivity: int,
    slab_depth: int,
    n_keep: int,
    logger: logging.Logger,
    on_slab: "Callable[[int, int, int, int, int], None] | None" = None,
):
    """
    on_slab: optional callback called after each slab completes.
    Signature: (pass_num, slab_idx, n_slabs, z0, z1) -> None
    Used by the Rich UI to show live slab progress.
    """
    if n_keep > 255:
        raise ValueError("n_keep must be <= 255 for uint8 output")

    z_dim, y_dim, x_dim = vol.shape
    cc_conn = CC3D_CONNECTIVITY[connectivity]
    n_slabs = int(np.ceil(z_dim / slab_depth))

    uf = UnionFind()
    provisional_sizes: dict[int, int] = {}
    mask_buf = np.empty((slab_depth + 2, y_dim, x_dim), dtype=bool)

    logger.info("CC pass 1/2 | connectivity=%s | slabs=%d", CONNECTIVITY_NAME[connectivity], n_slabs)

    base_id = 0
    prev_plane = None

    for slab_idx, z0 in enumerate(range(0, z_dim, slab_depth), start=1):
        z1 = min(z0 + slab_depth, z_dim)
        halo_top = 1 if z0 > 0 else 0
        halo_bottom = 1 if z1 < z_dim else 0
        slab_h = (z1 - z0) + halo_top + halo_bottom

        start = z0 - halo_top
        stop = start + slab_h
        np.equal(vol[start:stop], gas_label, out=mask_buf[:slab_h])

        labels_slab = cc3d.connected_components(mask_buf[:slab_h], connectivity=cc_conn)
        n_local = int(labels_slab.max())
        core = labels_slab[halo_top:halo_top + (z1 - z0)]

        if n_local > 0:
            # Use core.max() not labels_slab.max() — halo rows can contain
            # labels that inflate n_local far beyond what exists in core,
            # causing np.bincount to allocate gigabytes unnecessarily.
            n_core = int(core.max())
            counts = np.bincount(core.ravel(), minlength=n_core + 1)
            for local_id in range(1, counts.size):
                count = int(counts[local_id])
                if count:
                    provisional_sizes[base_id + local_id] = provisional_sizes.get(base_id + local_id, 0) + count

        if z0 > 0:
            first_plane = core[0].copy()
            if n_local > 0:
                first_plane[first_plane > 0] += base_id
            for a, b in seam_equivalences(prev_plane, first_plane):
                uf.union(a, b)

        prev_plane = core[-1].copy()
        if n_local > 0:
            prev_plane[prev_plane > 0] += base_id

        base_id += n_local
        del labels_slab
        gc.collect()

        logger.info("  slab %d/%d | z=%d-%d", slab_idx, n_slabs, z0, z1 - 1)
        if on_slab is not None:
            on_slab(1, slab_idx, n_slabs, z0, z1 - 1)

    if base_id == 0:
        logger.warning("No gas voxels found for %s", CONNECTIVITY_NAME[connectivity])
        return np.zeros((z_dim, y_dim, x_dim), dtype=np.uint8), []

    root_sizes: dict[int, int] = {}
    for pid, size in provisional_sizes.items():
        root = uf.find(pid)
        root_sizes[root] = root_sizes.get(root, 0) + size

    top_roots = sorted(root_sizes.items(), key=lambda kv: kv[1], reverse=True)[:n_keep]
    keep_roots = [root for root, _ in top_roots]
    root_to_final = {root: i + 1 for i, root in enumerate(keep_roots)}

    logger.info("CC pass 2/2 | connectivity=%s | writing top-%d labels", CONNECTIVITY_NAME[connectivity], n_keep)

    labels_out = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint8)
    base_id = 0

    for slab_idx, z0 in enumerate(range(0, z_dim, slab_depth), start=1):
        z1 = min(z0 + slab_depth, z_dim)
        halo_top = 1 if z0 > 0 else 0
        halo_bottom = 1 if z1 < z_dim else 0
        slab_h = (z1 - z0) + halo_top + halo_bottom

        start = z0 - halo_top
        stop = start + slab_h
        np.equal(vol[start:stop], gas_label, out=mask_buf[:slab_h])

        labels_slab = cc3d.connected_components(mask_buf[:slab_h], connectivity=cc_conn)
        n_local = int(labels_slab.max())
        core = labels_slab[halo_top:halo_top + (z1 - z0)]

        if n_local > 0:
            # Same fix: bound LUT by core.max() not labels_slab.max()
            n_core = int(core.max())
            lut = np.zeros(n_core + 1, dtype=np.uint8)
            for local_id in range(1, n_core + 1):
                lut[local_id] = root_to_final.get(uf.find(base_id + local_id), 0)
            labels_out[z0:z1] = lut[core]

        base_id += n_local
        del labels_slab
        gc.collect()

        logger.info("  slab %d/%d | z=%d-%d", slab_idx, n_slabs, z0, z1 - 1)
        if on_slab is not None:
            on_slab(2, slab_idx, n_slabs, z0, z1 - 1)

    report = [(root_to_final[root], root_sizes[root]) for root in keep_roots]
    report.sort(key=lambda x: x[0])
    return labels_out, report