from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile as tiff

from .config import Config, CONNECTIVITY_NAME
from .preprocessing import find_flow_crop_z, compute_cluster_cog
from .simulation_domains import make_absolute_domain, make_gas_domain, make_water_domain

if TYPE_CHECKING:
    from .db import PipelineDB


@dataclass
class FrozenBox:
    """Fixed box defined at timestep X (flow-cropped Z, full Y/X), applied to earlier scans."""
    track_id: int
    label_id_at_X: int
    z0: int
    z1: int          # half-open
    y0: int
    y1: int
    x0: int
    x1: int
    voxel_count_at_X: int
    connectivity: int


def define_frozen_boxes(
    labels_out: np.ndarray,
    report: list[tuple[int, int]],
    label_to_track: dict[int, int],
    connectivity: int,
    cfg: Config,
    logger: logging.Logger,
) -> list[FrozenBox]:
    """Define one FrozenBox per tracked cluster at timestep X."""
    from scipy import ndimage as ndi

    _, y_dim, x_dim = labels_out.shape
    objects = ndi.find_objects(labels_out)
    boxes = []

    for label_id, voxel_count in report:
        track_id = label_to_track.get(label_id)
        if track_id is None:
            logger.debug("label %d at timestep X has no track_id — skipping frozen box", label_id)
            continue

        sl = objects[label_id - 1] if label_id - 1 < len(objects) else None
        if sl is None:
            logger.warning("label %d not found in label volume at timestep X", label_id)
            continue

        zsl, _, _ = sl
        z0_bbox, z1_bbox = zsl.start, zsl.stop
        mask_z = (labels_out[z0_bbox:z1_bbox] == label_id)
        try:
            z_in, z_out = find_flow_crop_z(mask_z, threshold_fraction=cfg.inlet_outlet_threshold)
            z0 = z0_bbox + z_in
            z1 = z0_bbox + z_out
        except ValueError as exc:
            logger.warning("label %d at timestep X: flow crop failed (%s) — using bbox Z", label_id, exc)
            z0, z1 = z0_bbox, z1_bbox

        boxes.append(FrozenBox(
            track_id=track_id, label_id_at_X=label_id,
            z0=z0, z1=z1, y0=0, y1=y_dim, x0=0, x1=x_dim,
            voxel_count_at_X=voxel_count, connectivity=connectivity,
        ))
        logger.info("frozen box track %02d | z[%d-%d] y[%d-%d] x[%d-%d] | voxels_at_X=%d",
                    track_id, z0, z1 - 1, 0, y_dim - 1, 0, x_dim - 1, voxel_count)

    return boxes


def apply_frozen_boxes(
    vol: np.ndarray,
    boxes: list[FrozenBox],
    spacing: tuple[float, float, float] | None,
    out_dir: Path,
    file_stem: str,
    scan_index: int,
    connectivity: int,
    gas_label: int,
    run_id: str,
    db: "PipelineDB",
    logger: logging.Logger,
) -> dict[int, tuple]:
    """Apply frozen boxes to an earlier scan: crop, count gas/brine, save domains, insert rows.

    Returns track_id -> (gas_count, sw_local, cog_global_zyx, cluster_z_extent).
    """
    conn_name = CONNECTIVITY_NAME[connectivity]
    out_dir.mkdir(parents=True, exist_ok=True)
    voxel_volume = None if spacing is None else spacing[0] * spacing[1] * spacing[2]
    results: dict[int, tuple] = {}

    for box in boxes:
        if box.connectivity != connectivity:
            continue

        vol_slice = vol[box.z0:box.z1, box.y0:box.y1, box.x0:box.x1]

        vol_path = out_dir / f"track_{box.track_id:02d}_volume_{conn_name}.tiff"
        tiff.imwrite(vol_path, np.ascontiguousarray(vol_slice), photometric="minisblack")

        gas_mask = np.empty(vol_slice.shape, dtype=np.uint8)
        np.equal(vol_slice, gas_label, out=gas_mask)
        gas_count = int(gas_mask.sum())

        # True Z-extent at this scan = Z-slices in the box that contain gas.
        if gas_count > 0:
            z_idx = np.where(gas_mask.any(axis=(1, 2)))[0]
            cluster_z_extent = int(z_idx[-1] - z_idx[0] + 1)
        else:
            cluster_z_extent = 0

        mask_path = out_dir / f"track_{box.track_id:02d}_mask_{conn_name}.tiff"
        tiff.imwrite(mask_path, gas_mask, photometric="minisblack")

        brine_count = int(np.sum(vol_slice == 1))
        total_fluid = gas_count + brine_count
        sw_local = brine_count / total_fluid if total_fluid > 0 else float("nan")

        if gas_count > 0:
            cog_local = compute_cluster_cog(gas_mask)
            cog_global = (cog_local[0] + box.z0, cog_local[1] + box.y0, cog_local[2] + box.x0)
        else:
            cog_global = ((box.z0 + box.z1) / 2.0, (box.y0 + box.y1) / 2.0, (box.x0 + box.x1) / 2.0)

        results[box.track_id] = (gas_count, sw_local, cog_global, cluster_z_extent)

        abs_path = out_dir / f"track_{box.track_id:02d}_domain_absolute_{conn_name}.raw"
        gas_path = out_dir / f"track_{box.track_id:02d}_domain_gas_{conn_name}.raw"
        water_path = out_dir / f"track_{box.track_id:02d}_domain_water_{conn_name}.raw"
        make_absolute_domain(vol_slice).tofile(abs_path)
        make_gas_domain(vol_slice, gas_mask).tofile(gas_path)
        make_water_domain(vol_slice).tofile(water_path)
        del vol_slice

        row = {
            "file_stem":                file_stem,
            "connectivity":             conn_name,
            "track_id":                 box.track_id,
            "label_id_at_X":            box.label_id_at_X,
            "crop_mode":                "fixed",
            "z0":                       box.z0,
            "z1":                       box.z1 - 1,
            "y0":                       box.y0,
            "y1":                       box.y1 - 1,
            "x0":                       box.x0,
            "x1":                       box.x1 - 1,
            "extent_z":                 box.z1 - box.z0,
            "extent_y":                 box.y1 - box.y0,
            "extent_x":                 box.x1 - box.x0,
            "gas_voxels_this_timestep": gas_count,
            "gas_voxels_at_X":          box.voxel_count_at_X,
            "brine_voxels":             brine_count,
            "sw_local":                 sw_local,
            "volume_tiff":              str(vol_path),
            "mask_tiff":                str(mask_path),
            "domain_absolute":          str(abs_path),
            "domain_gas":               str(gas_path),
            "domain_water":             str(water_path),
        }
        if voxel_volume is not None:
            row["gas_volume_mm3"] = gas_count * voxel_volume * 1e9

        db.insert_fixed_box(run_id, scan_index, row)

        logger.info(
            "applied frozen box track %02d | gas=%d brine=%d sw_local=%.4f | cog=(%.1f,%.1f,%.1f) | file=%s",
            box.track_id, gas_count, brine_count,
            sw_local if sw_local == sw_local else float("nan"),
            cog_global[0], cog_global[1], cog_global[2], file_stem,
        )

    return results
