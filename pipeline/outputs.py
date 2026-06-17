from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi

from .config import CONNECTIVITY_NAME
from .preprocessing import find_flow_crop_z
from .simulation_domains import make_absolute_domain, make_gas_domain, make_water_domain

if TYPE_CHECKING:
    from .db import PipelineDB


def save_outputs(
    labels_out: np.ndarray,
    report: list[tuple[int, int]],
    vol: np.ndarray,
    spacing: tuple[float, float, float] | None,
    out_dir: Path,
    connectivity: int,
    crop_margin: int,
    inlet_outlet_threshold: float,
    label_to_track: dict[int, int],
    gas_label: int,
    scan_index: int,
    run_id: str,
    db: "PipelineDB",
    logger: logging.Logger,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn_name = CONNECTIVITY_NAME[connectivity]
    clustermask_info: dict = {}

    z_dim, y_dim, x_dim = labels_out.shape
    voxel_volume = None if spacing is None else spacing[0] * spacing[1] * spacing[2]
    objects = ndi.find_objects(labels_out)

    for label_id, voxel_count in report:
        sl = objects[label_id - 1] if label_id - 1 < len(objects) else None
        if sl is None:
            logger.warning("cluster %d not found in label volume", label_id)
            continue

        zsl, ysl, xsl = sl
        z0_bbox = max(zsl.start - crop_margin, 0)
        z1_bbox = min(zsl.stop + crop_margin, z_dim)
        y0, y1 = 0, y_dim
        x0, x1 = 0, x_dim

        mask_full = (labels_out[z0_bbox:z1_bbox] == label_id)
        try:
            z_in, z_out = find_flow_crop_z(mask_full, threshold_fraction=inlet_outlet_threshold)
            z0 = z0_bbox + z_in
            z1 = z0_bbox + z_out
        except ValueError as exc:
            logger.warning("cluster %02d: flow crop failed (%s) — using bbox crop", label_id, exc)
            z0, z1 = z0_bbox, z1_bbox

        vol_crop = np.array(vol[z0:z1, y0:y1, x0:x1])
        # mask_crop = (labels_out[z0:z1, y0:y1, x0:x1] == label_id).astype(np.uint8)

        # mask_path = out_dir / f"cluster_{label_id:02d}_mask_{conn_name}.tiff"
        # vol_path = out_dir / f"cluster_{label_id:02d}_volume_{conn_name}.tiff"
        # tiff.imwrite(mask_path, mask_crop, photometric="minisblack")
        # tiff.imwrite(vol_path, vol_crop, photometric="minisblack")

        all_gas_mask = (vol_crop == gas_label).astype(np.uint8)
        abs_path = out_dir / f"cluster_{label_id:02d}_domain_absolute_{conn_name}.raw"
        gas_path = out_dir / f"cluster_{label_id:02d}_domain_gas_{conn_name}.raw"
        water_path = out_dir / f"cluster_{label_id:02d}_domain_water_{conn_name}.raw"
        make_absolute_domain(vol_crop).tofile(abs_path)
        make_gas_domain(vol_crop, all_gas_mask).tofile(gas_path)
        make_water_domain(vol_crop).tofile(water_path)

        # Full tracked cluster mask over the cluster's OWN Z-extent (bbox, full
        # Y/X) — i.e. the whole connected cluster including the sub-threshold
        # tails the 10% flow-crop removes. Saved as 0/1 so the viewer can show
        # the whole cluster with the cropped extraction box drawn inside it.
        clustermask = (labels_out[z0_bbox:z1_bbox, y0:y1, x0:x1] == label_id).astype(np.uint8)
        clustermask_path = out_dir / f"cluster_{label_id:02d}_clustermask_{conn_name}.raw"
        clustermask.tofile(clustermask_path)
        cm_z0, cm_z1 = z0_bbox, z1_bbox - 1   # inclusive, full-volume coords

        track_id = label_to_track.get(label_id)
        row = {
            "connectivity":      conn_name,
            "label_id":          label_id,
            "track_id":          track_id,
            "voxel_count":       voxel_count,
            "crop_z_min_vox":    z0,
            "crop_z_max_vox":    z1 - 1,
            "crop_y_min_vox":    y0,
            "crop_y_max_vox":    y1 - 1,
            "crop_x_min_vox":    x0,
            "crop_x_max_vox":    x1 - 1,
            "crop_extent_z_vox": z1 - z0,
            "crop_extent_y_vox": y1 - y0,
            "crop_extent_x_vox": x1 - x0,
            "mask_tiff":         None,
            "volume_tiff":       None,
            "domain_absolute":   str(abs_path),
            "domain_gas":        str(gas_path),
            "domain_water":      str(water_path),
            "clustermask_raw":   str(clustermask_path),
            "clustermask_z0":    cm_z0,
            "clustermask_z1":    cm_z1,
        }
        if voxel_volume is not None:
            row["volume_mm3"] = voxel_count * voxel_volume * 1e9

        db.insert_cluster_property(run_id, scan_index, row)

        if track_id is not None:
            clustermask_info[track_id] = {
                "path": clustermask_path,
                "z0": cm_z0, "z1": cm_z1, "y0": y0, "x0": x0,
                "shape": (cm_z1 - cm_z0 + 1, y1 - y0, x1 - x0),
            }

        logger.info(
            "saved cluster %02d | track=%s | voxels=%d | bbox z[%d-%d] | flow crop z[%d-%d]",
            label_id, f"{track_id}" if track_id is not None else "none",
            voxel_count, zsl.start, zsl.stop - 1, z0, z1 - 1,
        )

    return clustermask_info
