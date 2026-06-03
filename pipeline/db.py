"""
pipeline/db.py

DuckDB interface for the pipeline.
One persistent database at the project root: results.duckdb
Each pipeline run creates one row in `runs` and inserts into all child tables.

Usage:
    db = PipelineDB(db_path=Path("results.duckdb"))
    db.init_run(run_id, cfg)
    db.insert_scan(run_id, scan_index, ...)
    db.insert_track(run_id, track_id, ...)
    db.insert_fixed_box(run_id, scan_index, ...)
    db.insert_cluster_property(run_id, scan_index, ...)
    db.close()

    # Export any table to CSV on demand
    db.export_csv("fixed_boxes", Path("export.csv"), run_id=run_id)
"""
from __future__ import annotations

import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from .config import Config


def generate_run_id() -> str:
    """Auto-generate a unique, human-readable run ID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    host = socket.gethostname().replace(" ", "_")[:16]
    return f"{ts}_{host}"


# ── Schema ──────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    started_at    TIMESTAMP,
    crop_mode     TEXT,
    connectivity  TEXT,
    n_keep        INTEGER,
    regime_cutoff TEXT,
    out_dir       TEXT
);

CREATE TABLE IF NOT EXISTS scans (
    run_id          TEXT    NOT NULL REFERENCES runs(run_id),
    scan_index      INTEGER NOT NULL,
    file_name       TEXT,
    Sw              DOUBLE,
    qualifying      BOOLEAN,
    is_X            BOOLEAN,
    elapsed_minutes DOUBLE,   -- from reference saturation file; NULL if not provided
    sw_ref          DOUBLE,   -- reference Sw (1 - Sg) from saturation file; NULL if not provided
    PRIMARY KEY (run_id, scan_index)
);

CREATE TABLE IF NOT EXISTS tracks (
    run_id          TEXT    NOT NULL REFERENCES runs(run_id),
    track_id        INTEGER NOT NULL,
    connectivity    TEXT    NOT NULL,
    status          TEXT,
    first_seen      TEXT,
    last_seen       TEXT,
    timesteps_seen  INTEGER,
    timesteps_lost  INTEGER,
    final_voxels    INTEGER,
    cog_z           DOUBLE,
    cog_y           DOUBLE,
    cog_x           DOUBLE,
    PRIMARY KEY (run_id, track_id, connectivity)
);

CREATE TABLE IF NOT EXISTS fixed_boxes (
    run_id                  TEXT    NOT NULL REFERENCES runs(run_id),
    scan_index              INTEGER NOT NULL,
    track_id                INTEGER NOT NULL,
    connectivity            TEXT    NOT NULL,
    z0                      INTEGER,
    z1                      INTEGER,
    y0                      INTEGER,
    y1                      INTEGER,
    x0                      INTEGER,
    x1                      INTEGER,
    extent_z                INTEGER,
    extent_y                INTEGER,
    extent_x                INTEGER,
    gas_voxels              INTEGER,
    gas_voxels_at_X         INTEGER,
    brine_voxels            INTEGER,
    gas_volume_mm3          DOUBLE,
    sw_local                DOUBLE,   -- section-level Sw = brine / (brine + gas) within this box
    volume_tiff             TEXT,
    mask_tiff               TEXT,
    domain_absolute         TEXT,
    domain_gas              TEXT,
    domain_water            TEXT,
    PRIMARY KEY (run_id, scan_index, track_id, connectivity)
);

CREATE TABLE IF NOT EXISTS cluster_properties (
    run_id          TEXT    NOT NULL REFERENCES runs(run_id),
    scan_index      INTEGER NOT NULL,
    label_id        INTEGER NOT NULL,
    track_id        INTEGER,
    connectivity    TEXT    NOT NULL,
    voxel_count     INTEGER,
    crop_z0         INTEGER,
    crop_z1         INTEGER,
    crop_y0         INTEGER,
    crop_y1         INTEGER,
    crop_x0         INTEGER,
    crop_x1         INTEGER,
    extent_z        INTEGER,
    extent_y        INTEGER,
    extent_x        INTEGER,
    volume_mm3      DOUBLE,
    volume_tiff     TEXT,
    mask_tiff       TEXT,
    domain_absolute TEXT,
    domain_gas      TEXT,
    domain_water    TEXT,
    PRIMARY KEY (run_id, scan_index, label_id, connectivity)
);

CREATE TABLE IF NOT EXISTS simulation_results (
    run_id          TEXT    NOT NULL REFERENCES runs(run_id),
    scan_index      INTEGER NOT NULL,
    track_id        INTEGER NOT NULL,
    connectivity    TEXT    NOT NULL,
    sim_type        TEXT    NOT NULL,
    simulator       TEXT    NOT NULL,
    k_x             DOUBLE,
    k_y             DOUBLE,
    k_z             DOUBLE,
    k_eff           DOUBLE,
    kr              DOUBLE,
    Sw              DOUBLE,
    domain_path     TEXT,
    raw_output_path TEXT,
    notes           TEXT,
    PRIMARY KEY (run_id, scan_index, track_id, connectivity, sim_type, simulator)
);
"""


class PipelineDB:
    """
    Thin wrapper around a DuckDB connection.
    All write methods use parameterised queries — no string interpolation.
    """

    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(db_path))
        self._con.execute(_SCHEMA)
        self._db_path = db_path

    # ── Run ─────────────────────────────────────────────────────────────────

    def init_run(self, run_id: str, cfg: Config) -> None:
        self._con.execute(
            """
            INSERT OR IGNORE INTO runs
                (run_id, started_at, crop_mode, connectivity, n_keep,
                 regime_cutoff, out_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                datetime.now(),
                cfg.crop_mode,
                ",".join(str(c) for c in cfg.connectivities),
                cfg.n_keep,
                cfg.regime_cutoff,
                str(cfg.out_dir),
            ),
        )

    # ── Scans ────────────────────────────────────────────────────────────────

    def insert_scan(
        self,
        run_id: str,
        scan_index: int,
        file_name: str,
        sw: float,
        qualifying: bool,
        is_X: bool,
        elapsed_minutes: float | None = None,
        sw_ref: float | None = None,
    ) -> None:
        self._con.execute(
            """
            INSERT OR REPLACE INTO scans
                (run_id, scan_index, file_name, Sw, qualifying, is_X,
                 elapsed_minutes, sw_ref)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, scan_index, file_name, sw, qualifying, is_X,
             elapsed_minutes, sw_ref),
        )

    def insert_sw_series(
        self,
        run_id: str,
        sw_series: list[float],
        all_files: list[Path],
        X: int,
        elapsed_minutes: list[float] | None = None,
        sat_records: dict | None = None,
    ) -> None:
        """
        Batch insert all scan rows at once after Stage 1.

        elapsed_minutes: list aligned to all_files; NaN where unavailable.
        sat_records: dict of scan_index -> SaturationRecord for sw_ref values.
        """
        rows = []
        for i, f in enumerate(all_files):
            em = None
            if elapsed_minutes is not None and i < len(elapsed_minutes):
                v = elapsed_minutes[i]
                em = None if (v != v) else v  # NaN -> None
            sr = None
            if sat_records is not None and i in sat_records:
                sr = sat_records[i].sw_ref
            rows.append((
                run_id, i, f.name, sw_series[i],
                i <= X, i == X, em, sr,
            ))
        self._con.executemany(
            """
            INSERT OR REPLACE INTO scans
                (run_id, scan_index, file_name, Sw, qualifying, is_X,
                 elapsed_minutes, sw_ref)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    # ── Tracks ───────────────────────────────────────────────────────────────

    def insert_track(self, run_id: str, connectivity: str, row: dict) -> None:
        self._con.execute(
            """
            INSERT OR REPLACE INTO tracks
                (run_id, track_id, connectivity, status,
                 first_seen, last_seen, timesteps_seen, timesteps_lost,
                 final_voxels, cog_z, cog_y, cog_x)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                row["track_id"],
                connectivity,
                row["status"],
                row["first_seen"],
                row["last_seen"],
                row["timesteps_seen"],
                row["timesteps_lost"],
                row["final_voxel_count"],
                row["final_cog_z"],
                row["final_cog_y"],
                row["final_cog_x"],
            ),
        )

    def insert_tracks_bulk(
        self, run_id: str, connectivity: str, rows: list[dict]
    ) -> None:
        self._con.executemany(
            """
            INSERT OR REPLACE INTO tracks
                (run_id, track_id, connectivity, status,
                 first_seen, last_seen, timesteps_seen, timesteps_lost,
                 final_voxels, cog_z, cog_y, cog_x)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    r["track_id"],
                    connectivity,
                    r["status"],
                    r["first_seen"],
                    r["last_seen"],
                    r["timesteps_seen"],
                    r["timesteps_lost"],
                    r["final_voxel_count"],
                    r["final_cog_z"],
                    r["final_cog_y"],
                    r["final_cog_x"],
                )
                for r in rows
            ],
        )

    # ── Fixed boxes ──────────────────────────────────────────────────────────

    def insert_fixed_box(self, run_id: str, scan_index: int, row: dict) -> None:
        self._con.execute(
            """
            INSERT OR REPLACE INTO fixed_boxes
                (run_id, scan_index, track_id, connectivity,
                 z0, z1, y0, y1, x0, x1,
                 extent_z, extent_y, extent_x,
                 gas_voxels, gas_voxels_at_X, brine_voxels, gas_volume_mm3, sw_local,
                 volume_tiff, mask_tiff,
                 domain_absolute, domain_gas, domain_water)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                scan_index,
                row["track_id"],
                row["connectivity"],
                row["z0"], row["z1"],
                row["y0"], row["y1"],
                row["x0"], row["x1"],
                row["extent_z"], row["extent_y"], row["extent_x"],
                row["gas_voxels_this_timestep"],
                row["gas_voxels_at_X"],
                row.get("brine_voxels"),
                row.get("gas_volume_mm3"),
                row.get("sw_local"),
                row["volume_tiff"],
                row["mask_tiff"],
                row["domain_absolute"],
                row["domain_gas"],
                row["domain_water"],
            ),
        )

    def update_fixed_box_sw(
        self,
        run_id: str,
        scan_index: int,
        track_id: int,
        connectivity: str,
        sw_local: float,
        brine_voxels: int,
    ) -> None:
        """
        Back-fill sw_local and brine_voxels on a fixed_box row that was already
        inserted (used for timestep X, where the row is written by insert_fixed_box
        via cluster_properties path and sw is computed afterwards from the volume).
        """
        self._con.execute(
            """
            UPDATE fixed_boxes
               SET sw_local = ?, brine_voxels = ?
             WHERE run_id = ? AND scan_index = ? AND track_id = ? AND connectivity = ?
            """,
            (sw_local, brine_voxels, run_id, scan_index, track_id, connectivity),
        )

    # ── Cluster properties ───────────────────────────────────────────────────

    def insert_cluster_property(
        self, run_id: str, scan_index: int, row: dict
    ) -> None:
        self._con.execute(
            """
            INSERT OR REPLACE INTO cluster_properties
                (run_id, scan_index, label_id, track_id, connectivity,
                 voxel_count,
                 crop_z0, crop_z1, crop_y0, crop_y1, crop_x0, crop_x1,
                 extent_z, extent_y, extent_x,
                 volume_mm3,
                 volume_tiff, mask_tiff,
                 domain_absolute, domain_gas, domain_water)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                scan_index,
                row["label_id"],
                row.get("track_id"),
                row["connectivity"],
                row["voxel_count"],
                row["crop_z_min_vox"], row["crop_z_max_vox"],
                row["crop_y_min_vox"], row["crop_y_max_vox"],
                row["crop_x_min_vox"], row["crop_x_max_vox"],
                row["crop_extent_z_vox"],
                row["crop_extent_y_vox"],
                row["crop_extent_x_vox"],
                row.get("volume_mm3"),
                row["volume_tiff"],
                row["mask_tiff"],
                row["domain_absolute"],
                row["domain_gas"],
                row["domain_water"],
            ),
        )

    # ── Export ───────────────────────────────────────────────────────────────

    def export_csv(
        self,
        table: str,
        path: Path,
        run_id: str | None = None,
    ) -> None:
        """
        Export a table (or filtered view) to CSV.

        Example:
            db.export_csv("fixed_boxes", Path("fixed_boxes_run1.csv"), run_id=run_id)
        """
        valid_tables = {
            "runs", "scans", "tracks",
            "fixed_boxes", "cluster_properties", "simulation_results",
        }
        if table not in valid_tables:
            raise ValueError(f"Unknown table: {table!r}. Valid: {valid_tables}")

        path = Path(path)
        if run_id:
            query = f"SELECT * FROM {table} WHERE run_id = '{run_id}'"
        else:
            query = f"SELECT * FROM {table}"

        self._con.execute(
            f"COPY ({query}) TO '{path}' (HEADER, DELIMITER ',')"
        )

    def export_all_csv(self, out_dir: Path, run_id: str) -> None:
        """Export all tables for a given run to CSV files in out_dir."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for table in ("scans", "tracks", "fixed_boxes",
                      "cluster_properties", "simulation_results"):
            self.export_csv(table, out_dir / f"{table}.csv", run_id=run_id)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()