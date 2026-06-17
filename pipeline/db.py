from __future__ import annotations

import socket
from datetime import datetime
from pathlib import Path

import duckdb

from .config import Config


def generate_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    host = socket.gethostname().replace(" ", "_")[:16]
    return f"{ts}_{host}"


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
    elapsed_minutes DOUBLE,
    sw_ref          DOUBLE,
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
    sw_local                DOUBLE,
    percolates              BOOLEAN,  -- a gas component spans box inlet-to-outlet in Z
    spanning_count          INTEGER,  -- how many components span (>1 = the cluster split)
    cluster_voxels          INTEGER,  -- voxels in the largest spanning (percolating) cluster
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
    clustermask_raw TEXT,     -- full tracked cluster (0/1) over its own Z-extent, full Y/X
    clustermask_z0  INTEGER,  -- cluster mask first Z-slice (full-volume coords)
    clustermask_z1  INTEGER,  -- cluster mask last Z-slice (inclusive)
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

CREATE TABLE IF NOT EXISTS prior_work_provenance (
    run_id          TEXT    NOT NULL REFERENCES runs(run_id),
    scan_file       TEXT    NOT NULL,   -- the .am file this step came from
    step_order      INTEGER NOT NULL,   -- order within that file's HistoryLog
    module          TEXT,               -- Avizo module name (e.g. InteractiveThresholding)
    label           TEXT,               -- human label (e.g. "Interactive Thresholding")
    avizo_version   TEXT,
    step_date       TEXT,               -- ISO date Avizo recorded for the step
    parameters      TEXT,               -- full parameter set as JSON (verbatim)
    lattice_z       INTEGER,            -- volume dims parsed from 'define Lattice'
    lattice_y       INTEGER,
    lattice_x       INTEGER,
    bbox            TEXT,               -- BoundingBox (xmin xmax ymin ymax zmin zmax) as text
    voxel_dz_m      DOUBLE,             -- voxel size (metres) extracted from bbox/(n-1)
    voxel_dy_m      DOUBLE,
    voxel_dx_m      DOUBLE,
    PRIMARY KEY (run_id, scan_file, step_order)
);
"""


class PipelineDB:
    """Thin DuckDB wrapper; all writes use parameterised queries."""

    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(db_path))
        self._con.execute(_SCHEMA)
        self._migrate()
        self._db_path = db_path

    def _migrate(self) -> None:
        """Add columns introduced after a DB was first created. Idempotent."""
        added = {
            "fixed_boxes": [
                ("percolates", "BOOLEAN"),
                ("spanning_count", "INTEGER"),
                ("cluster_voxels", "INTEGER"),
            ],
            "cluster_properties": [
                ("clustermask_raw", "TEXT"),
                ("clustermask_z0", "INTEGER"),
                ("clustermask_z1", "INTEGER"),
            ],
        }
        for table, cols in added.items():
            existing = {r[0] for r in self._con.execute(
                "SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{table}'").fetchall()}
            for name, sqltype in cols:
                if name not in existing:
                    self._con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}")

    def init_run(self, run_id: str, cfg: Config) -> None:
        self._con.execute(
            """
            INSERT OR IGNORE INTO runs
                (run_id, started_at, crop_mode, connectivity, n_keep, regime_cutoff, out_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, datetime.now(), cfg.crop_mode,
             ",".join(str(c) for c in cfg.connectivities),
             cfg.n_keep, cfg.regime_cutoff, str(cfg.out_dir)),
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
        rows = []
        for i, f in enumerate(all_files):
            em = None
            if elapsed_minutes is not None and i < len(elapsed_minutes):
                v = elapsed_minutes[i]
                em = None if (v != v) else v  # NaN -> None
            sr = sat_records[i].sw_ref if (sat_records is not None and i in sat_records) else None
            rows.append((run_id, i, f.name, sw_series[i], i <= X, i == X, em, sr))
        self._con.executemany(
            """
            INSERT OR REPLACE INTO scans
                (run_id, scan_index, file_name, Sw, qualifying, is_X, elapsed_minutes, sw_ref)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def insert_tracks_bulk(self, run_id: str, connectivity: str, rows: list[dict]) -> None:
        self._con.executemany(
            """
            INSERT OR REPLACE INTO tracks
                (run_id, track_id, connectivity, status, first_seen, last_seen,
                 timesteps_seen, timesteps_lost, final_voxels, cog_z, cog_y, cog_x)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (run_id, r["track_id"], connectivity, r["status"], r["first_seen"], r["last_seen"],
                 r["timesteps_seen"], r["timesteps_lost"], r["final_voxel_count"],
                 r["final_cog_z"], r["final_cog_y"], r["final_cog_x"])
                for r in rows
            ],
        )

    def insert_fixed_box(self, run_id: str, scan_index: int, row: dict) -> None:
        self._con.execute(
            """
            INSERT OR REPLACE INTO fixed_boxes
                (run_id, scan_index, track_id, connectivity,
                 z0, z1, y0, y1, x0, x1, extent_z, extent_y, extent_x,
                 gas_voxels, gas_voxels_at_X, brine_voxels, gas_volume_mm3, sw_local,
                 percolates, spanning_count, cluster_voxels,
                 volume_tiff, mask_tiff, domain_absolute, domain_gas, domain_water)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, scan_index, row["track_id"], row["connectivity"],
             row["z0"], row["z1"], row["y0"], row["y1"], row["x0"], row["x1"],
             row["extent_z"], row["extent_y"], row["extent_x"],
             row["gas_voxels_this_timestep"], row["gas_voxels_at_X"],
             row.get("brine_voxels"), row.get("gas_volume_mm3"), row.get("sw_local"),
             row.get("percolates"), row.get("spanning_count"), row.get("cluster_voxels"),
             row["volume_tiff"], row["mask_tiff"],
             row["domain_absolute"], row["domain_gas"], row["domain_water"]),
        )

    def insert_cluster_property(self, run_id: str, scan_index: int, row: dict) -> None:
        self._con.execute(
            """
            INSERT OR REPLACE INTO cluster_properties
                (run_id, scan_index, label_id, track_id, connectivity, voxel_count,
                 crop_z0, crop_z1, crop_y0, crop_y1, crop_x0, crop_x1,
                 extent_z, extent_y, extent_x, volume_mm3,
                 volume_tiff, mask_tiff, domain_absolute, domain_gas, domain_water,
                 clustermask_raw, clustermask_z0, clustermask_z1)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, scan_index, row["label_id"], row.get("track_id"), row["connectivity"],
             row["voxel_count"],
             row["crop_z_min_vox"], row["crop_z_max_vox"],
             row["crop_y_min_vox"], row["crop_y_max_vox"],
             row["crop_x_min_vox"], row["crop_x_max_vox"],
             row["crop_extent_z_vox"], row["crop_extent_y_vox"], row["crop_extent_x_vox"],
             row.get("volume_mm3"),
             row["volume_tiff"], row["mask_tiff"],
             row["domain_absolute"], row["domain_gas"], row["domain_water"],
             row.get("clustermask_raw"), row.get("clustermask_z0"), row.get("clustermask_z1")),
        )

    def insert_prior_work_provenance(
        self, run_id: str, scan_file: str, steps: list[dict],
        lattice=None, bbox=None, voxel_size=None,
    ) -> None:
        """Store the parsed Avizo HistoryLog chain for one .am file. Idempotent
        per (run_id, scan_file): existing rows for this file are replaced.
        voxel_size is (dz, dy, dx) in metres, extracted from bbox/(n-1)."""
        import json
        lz, ly, lx = (lattice if lattice else (None, None, None))
        bbox_txt = " ".join(f"{v:.10g}" for v in bbox) if bbox else None
        vdz, vdy, vdx = (voxel_size if voxel_size else (None, None, None))
        self._con.execute(
            "DELETE FROM prior_work_provenance WHERE run_id = ? AND scan_file = ?",
            (run_id, scan_file),
        )
        for s in steps:
            self._con.execute(
                """
                INSERT INTO prior_work_provenance
                    (run_id, scan_file, step_order, module, label,
                     avizo_version, step_date, parameters,
                     lattice_z, lattice_y, lattice_x, bbox,
                     voxel_dz_m, voxel_dy_m, voxel_dx_m)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, scan_file, s["order"], s.get("module"), s.get("label"),
                 s.get("avizo_version"), s.get("date"),
                 json.dumps(s.get("parameters", {})),
                 lz, ly, lx, bbox_txt, vdz, vdy, vdx),
            )

    def update_percolation(
        self, run_id: str, scan_index: int, track_id: int, connectivity: str,
        percolates: bool, spanning_count: int, cluster_voxels: int,
    ) -> None:
        """Write percolation results into an existing fixed_boxes row. Called by
        the viewer's "Check percolation" button after it runs cc3d for a
        (track, scan)."""
        self._con.execute(
            """
            UPDATE fixed_boxes
               SET percolates = ?, spanning_count = ?, cluster_voxels = ?
             WHERE run_id = ? AND scan_index = ? AND track_id = ? AND connectivity = ?
            """,
            (percolates, spanning_count, cluster_voxels,
             run_id, scan_index, track_id, connectivity),
        )

    def close(self) -> None:
        self._con.close()