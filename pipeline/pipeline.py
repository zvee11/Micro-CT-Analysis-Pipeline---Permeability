from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .config import Config, CONNECTIVITY_NAME
from .connected import topn_gas_cc
from .db import PipelineDB, generate_run_id
from .fixed_box import FrozenBox, apply_frozen_boxes, define_frozen_boxes
from .io import read_avizo, iter_input_files
from .outputs import save_outputs
from .preprocessing import compute_cluster_cog, detect_and_fix_x_wraparound, log_volume_info
from .regime import compute_sw_series, detect_regime_boundary
from .saturation import load_saturation_reference, build_elapsed_minutes_array
from .tracking import ClusterTracker


def setup_logging(log_path: Path) -> logging.Logger:
    """Send detailed logs to a file; the terminal is driven by the Rich UI."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="w")],
        force=True,
    )
    return logging.getLogger("pore_time_baby")


def _load_volume(path, logger: logging.Logger):
    vol, spacing, info = read_avizo(path, parse_spacing=True, memmap_raw=False)
    vol = detect_and_fix_x_wraparound(vol, logger)
    return vol, spacing, info


def _run_cc(vol, spacing, cfg, input_path, connectivity, logger, tracker,
            is_first, is_last, ui=None):
    """Run connected-components on one volume. Returns (labels_out, report, label_to_track)."""
    conn_name = CONNECTIVITY_NAME[connectivity]

    def _on_slab(pass_num, slab_idx, n_slabs, z0, z1):
        if ui is not None:
            ui.update_slab(pass_num, slab_idx, n_slabs, z0, z1, conn_name)

    labels_out, report = topn_gas_cc(
        vol=vol, gas_label=cfg.gas_label, connectivity=connectivity,
        slab_depth=cfg.slab_depth, n_keep=cfg.n_keep, logger=logger, on_slab=_on_slab,
    )

    if not report:
        logger.warning("no clusters found for %s | file=%s", conn_name, input_path.name)
        if tracker is not None:
            tracker.update(input_path.stem, [], logger, is_first=is_first, is_last=is_last)
        return None, [], {}

    label_to_track: dict[int, int] = {}
    if tracker is not None:
        objects = ndi.find_objects(labels_out)
        cluster_info = []
        for label_id, voxel_count in report:
            sl = objects[label_id - 1] if label_id - 1 < len(objects) else None
            if sl is None:
                continue
            cog = compute_cluster_cog(labels_out == label_id)
            cluster_info.append((label_id, cog, voxel_count))
        label_to_track = tracker.update(input_path.stem, cluster_info, logger,
                                        is_first=is_first, is_last=is_last)

    return labels_out, report, label_to_track


def main() -> None:
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(cfg.out_dir / "pipeline.log")
    run_id = generate_run_id()
    logger.info("run_id: %s", run_id)

    all_files = iter_input_files(cfg)
    n_total = len(all_files)
    logger.info("found %d input files", n_total)

    db = PipelineDB(db_path=Path("results.duckdb"))
    db.init_run(run_id, cfg)

    # ── Optional UI / visualisers ───────────────────────────────────────────
    ui = None
    if cfg.enable_ui:
        try:
            from .ui import PipelineUI
            ui = PipelineUI(n_total=n_total, crop_mode=cfg.crop_mode, run_name=run_id)
            ui.start()
        except ImportError:
            pass

    pv_vis = None
    if cfg.enable_pyvista:
        try:
            from .visualisation import PyVistaVisualiser
            pv_vis = PyVistaVisualiser(downsample=cfg.pyvista_downsample)
        except ImportError:
            pass

    dash_vis = None
    if cfg.enable_dash:
        try:
            from .visualisation import DashVisualiser
            dash_vis = DashVisualiser(out_dir=cfg.out_dir, port=cfg.dash_port)
        except ImportError:
            pass

    # ── Stage 1: Sw series + regime detection ───────────────────────────────
    logger.info("STAGE 1 — pre-pass: computing Sw series")

    def _on_file_start(scan_idx, file_name):
        if ui:
            ui.prepass_file_start(scan_idx, file_name)

    def _on_file_done(scan_idx, file_name, sw):
        if ui:
            ui.prepass_file_done(scan_idx, file_name, sw)

    sw_series, hist_series = compute_sw_series(
        all_files, cfg, logger, on_file_start=_on_file_start, on_file_done=_on_file_done,
    )

    sat_records: dict = {}
    elapsed_minutes: list[float] = []
    if cfg.saturation_file:
        sat_records = load_saturation_reference(
            saturation_file=cfg.saturation_file, all_files=all_files,
            name_col=cfg.saturation_name_col, sg_col=cfg.saturation_sg_col,
            time_col=cfg.saturation_time_col, logger=logger,
        )
        elapsed_minutes = build_elapsed_minutes_array(all_files, sat_records)
        logger.info("saturation reference: %d/%d scans matched", len(sat_records), n_total)
    else:
        logger.info("no saturation_file configured — using scan index for regime detection")

    x_values = elapsed_minutes if elapsed_minutes else None
    X = detect_regime_boundary(sw_series, cfg, logger, x_values=x_values)
    qualifying_files = [all_files[i] for i in range(n_total) if i <= X]
    n_q = len(qualifying_files)
    file_to_scan_idx = {f: i for i, f in enumerate(all_files)}
    logger.info("qualifying timesteps: %d/%d (indices 0-%d)", n_q, n_total, X)

    db.insert_sw_series(
        run_id, sw_series, all_files, X,
        elapsed_minutes=elapsed_minutes if elapsed_minutes else None,
        sat_records=sat_records if sat_records else None,
    )

    if ui:
        ui.finish_prepass(X, n_q, sw_series)

    if dash_vis:
        dash_vis.init_sw_series(
            sw_series, X, all_files,
            elapsed_minutes=elapsed_minutes if elapsed_minutes else None,
            sat_records=sat_records if sat_records else None,
        )
        dash_vis.launch()

    # ── Stage 2: main pass (fixed mode) ─────────────────────────────────────
    logger.info("STAGE 2 — main pass | crop_mode=%s", cfg.crop_mode)

    trackers: dict[int, ClusterTracker] = {}
    if cfg.track_clusters:
        for connectivity in cfg.connectivities:
            trackers[connectivity] = ClusterTracker(n_keep=cfg.n_keep)

    frozen_boxes: dict[int, list[FrozenBox]] = {c: [] for c in cfg.connectivities}
    total_t0 = time.time()

    # ── Step A: timestep X — full CC, define frozen boxes ───────────────────
    x_path = qualifying_files[-1]
    logger.info("FIXED MODE — Step A: timestep X | %s", x_path.name)
    if ui:
        ui.start_step_a(x_path.name)

    t0 = time.time()
    vol, spacing, info = _load_volume(x_path, logger)
    log_volume_info(vol, spacing, info, logger, hist=hist_series[X])

    for connectivity in cfg.connectivities:
        labels_out, report, label_to_track = _run_cc(
            vol, spacing, cfg, x_path, connectivity, logger,
            trackers.get(connectivity), is_first=(n_q == 1), is_last=True, ui=ui,
        )
        if labels_out is None:
            continue

        conn_name_x = CONNECTIVITY_NAME[connectivity]
        out_dir_x = cfg.out_dir / x_path.stem / conn_name_x

        save_outputs(
            labels_out=labels_out, report=report, vol=vol, spacing=spacing,
            out_dir=out_dir_x, connectivity=connectivity, crop_margin=cfg.crop_margin,
            inlet_outlet_threshold=cfg.inlet_outlet_threshold, label_to_track=label_to_track,
            gas_label=cfg.gas_label, scan_index=X, run_id=run_id, db=db, logger=logger,
        )

        frozen_boxes[connectivity] = define_frozen_boxes(
            labels_out=labels_out, report=report, label_to_track=label_to_track,
            connectivity=connectivity, cfg=cfg, logger=logger,
        )

        # fixed_boxes is the downstream table for k_r; scan X needs its own rows here.
        # save_outputs only wrote cluster_properties, and vol is still loaded, so
        # compute brine count and sw_local now.
        voxel_volume_x = None if spacing is None else spacing[0] * spacing[1] * spacing[2]
        for box in frozen_boxes[connectivity]:
            vol_slice_x = vol[box.z0:box.z1, box.y0:box.y1, box.x0:box.x1]
            brine_x = int(np.sum(vol_slice_x == 1))
            total_x = box.voxel_count_at_X + brine_x
            sw_x = brine_x / total_x if total_x > 0 else float("nan")

            row_x = {
                "file_stem":                x_path.stem,
                "connectivity":             conn_name_x,
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
                "gas_voxels_this_timestep": box.voxel_count_at_X,
                "gas_voxels_at_X":          box.voxel_count_at_X,
                "brine_voxels":             brine_x,
                "sw_local":                 sw_x,
                "volume_tiff": str(out_dir_x / f"cluster_{box.label_id_at_X:02d}_volume_{conn_name_x}.tiff"),
                "mask_tiff":   str(out_dir_x / f"cluster_{box.label_id_at_X:02d}_mask_{conn_name_x}.tiff"),
                "domain_absolute": str(out_dir_x / f"cluster_{box.label_id_at_X:02d}_domain_absolute_{conn_name_x}.raw"),
                "domain_gas":      str(out_dir_x / f"cluster_{box.label_id_at_X:02d}_domain_gas_{conn_name_x}.raw"),
                "domain_water":    str(out_dir_x / f"cluster_{box.label_id_at_X:02d}_domain_water_{conn_name_x}.raw"),
            }
            if voxel_volume_x is not None:
                row_x["gas_volume_mm3"] = box.voxel_count_at_X * voxel_volume_x * 1e9
            db.insert_fixed_box(run_id, X, row_x)
            logger.info("timestep X fixed_box track %02d | brine=%d gas=%d sw_local=%.4f",
                        box.track_id, brine_x, box.voxel_count_at_X, sw_x)

        if dash_vis:
            for box in frozen_boxes[connectivity]:
                dash_vis.update_fixed_box(
                    scan_index=X, track_id=box.track_id, gas_voxels=box.voxel_count_at_X,
                    z_extent=box.z1 - box.z0, box_z_extent=box.z1 - box.z0,
                )

        if pv_vis is not None:
            for box in frozen_boxes[connectivity]:
                gas_path = (out_dir_x / f"cluster_{box.label_id_at_X:02d}_domain_gas_{conn_name_x}.raw")
                shape = (box.z1 - box.z0, box.y1 - box.y0, box.x1 - box.x0)
                pv_vis.register_cluster_at_X(
                    track_id=box.track_id, gas_domain_path=gas_path, shape=shape,
                    scan_index=X, spacing=spacing, origin=(box.z0, box.y0, box.x0),
                )

        del labels_out
        gc.collect()

    del vol
    gc.collect()

    elapsed_a = time.time() - t0
    logger.info("timestep X done in %.1fs — %d frozen boxes defined",
                elapsed_a, sum(len(b) for b in frozen_boxes.values()))
    if ui:
        all_boxes = [b for bl in frozen_boxes.values() for b in bl]
        ui.finish_step_a(all_boxes, elapsed_a)

    # ── Step B: earlier timesteps, reverse order ────────────────────────────
    logger.info("FIXED MODE — Step B: applying frozen boxes backwards")
    earlier_files = qualifying_files[:-1]

    for file_idx, input_path in enumerate(reversed(earlier_files), start=1):
        t0 = time.time()
        is_first = (file_idx == len(earlier_files))
        scan_idx = file_to_scan_idx[input_path]
        logger.info("Step B %d/%d | %s", file_idx, len(earlier_files), input_path.name)

        if ui:
            ui.start_step_b(file_idx, len(earlier_files), input_path.name)
            ui.show_current_file(
                f"Step B {file_idx}/{len(earlier_files)}", input_path.name,
                f"applying {len(frozen_boxes.get(cfg.connectivities[0], []))} frozen boxes",
            )

        vol, spacing, _ = _load_volume(input_path, logger)

        for connectivity in cfg.connectivities:
            boxes = frozen_boxes.get(connectivity, [])
            if not boxes:
                logger.warning("no frozen boxes for %s — skipping %s",
                               CONNECTIVITY_NAME[connectivity], input_path.name)
                continue

            box_results = apply_frozen_boxes(
                vol=vol, boxes=boxes, spacing=spacing,
                out_dir=cfg.out_dir / input_path.stem / CONNECTIVITY_NAME[connectivity],
                file_stem=input_path.stem, scan_index=scan_idx, connectivity=connectivity,
                gas_label=cfg.gas_label, run_id=run_id, db=db, logger=logger,
            )

            tracker = trackers.get(connectivity)
            if tracker is not None:
                cluster_info = [
                    (box.label_id_at_X, box_results[box.track_id][2], box_results[box.track_id][0])
                    for box in boxes if box.track_id in box_results
                ]
                tracker.update(input_path.stem, cluster_info, logger, is_first=is_first, is_last=False)

            if dash_vis:
                for box in boxes:
                    if box.track_id in box_results:
                        gas_count, _sw, _cog, cluster_zext = box_results[box.track_id]
                        dash_vis.update_fixed_box(
                            scan_index=scan_idx, track_id=box.track_id, gas_voxels=gas_count,
                            z_extent=cluster_zext, box_z_extent=box.z1 - box.z0,
                        )

        if pv_vis is not None:
            for connectivity in cfg.connectivities:
                conn_name = CONNECTIVITY_NAME[connectivity]
                for box in frozen_boxes.get(connectivity, []):
                    gas_path = (cfg.out_dir / input_path.stem / conn_name
                                / f"track_{box.track_id:02d}_domain_gas_{conn_name}.raw")
                    shape = (box.z1 - box.z0, box.y1 - box.y0, box.x1 - box.x0)
                    pv_vis.register_cluster_at_scan(
                        track_id=box.track_id, gas_domain_path=gas_path,
                        shape=shape, scan_index=scan_idx,
                    )

        del vol
        gc.collect()

        elapsed = time.time() - t0
        logger.info("finished %s in %.1fs", input_path.name, elapsed)
        if ui:
            ui.finish_file(input_path.name, elapsed)

    # ── Finalise ────────────────────────────────────────────────────────────
    if cfg.track_clusters:
        for connectivity, tracker in trackers.items():
            conn_name = CONNECTIVITY_NAME[connectivity]
            summary = tracker.summary()
            if summary:
                db.insert_tracks_bulk(run_id, conn_name, summary)
                logger.info("inserted %d track rows for %s", len(summary), conn_name)

    total_elapsed = time.time() - total_t0
    logger.info("all files done in %.1fs", total_elapsed)

    if ui:
        ui.finish(total_elapsed, sw_series, X)
        ui.stop()

    db.close()
    logger.info("database closed — run_id: %s", run_id)

    if pv_vis is not None:
        try:
            pv_vis.show_final()
        except Exception as e:
            logger.warning("PyVista viewer failed: %s", e)
