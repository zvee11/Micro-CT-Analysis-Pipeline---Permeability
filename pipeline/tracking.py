from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackedCluster:
    """
    State for a single cluster being tracked across timesteps.

    `track_id` is a stable identifier assigned at the timestep the cluster
    was first seen. It never changes even as label_id may differ across files.

    `cog` is stored as np.ndarray (shape (3,), float64) rather than a tuple
    so that distance calculations in _match() never allocate a temporary array
    — the stored value is already in the form numpy needs.
    """
    track_id: int
    first_seen: str
    last_seen: str
    cog: np.ndarray          # shape (3,) float64 — (z, y, x) in voxel coords
    voxel_count: int
    status: str = "active"   # "active" | "lost"
    timesteps_seen: int = 1
    timesteps_lost: int = 0


@dataclass
class ClusterTracker:
    """
    Tracks up to 2*N clusters across timesteps using centre-of-gravity matching.

    Seeding rules:
    - Top-N clusters from the FIRST processed timestep are always seeded.
    - Top-N clusters from the LAST processed timestep are always seeded
      (any not already matched get new track IDs).

    Matching (per timestep):
    - Incoming CoGs are converted to np.ndarray once at the top of update().
    - Each incoming cluster is matched to the nearest active track by
      Euclidean distance. O(n_clusters * n_tracks) — at most N * 2N = 2N²
      distance calculations, negligible for N <= 10.
    - If no active track is within max_cog_distance, the cluster is unmatched.
      Unmatched clusters are seeded only on first/last timestep.

    Lost tracks:
    - Any active track not matched in a timestep is marked "lost" and flagged
      in both the log and the final tracking CSV.
    """

    n_keep: int
    max_cog_distance: float = 1000.0
    _tracks: dict[int, TrackedCluster] = field(default_factory=dict)
    _next_id: int = field(default=1)
    _timestep_count: int = field(default=0)

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def update(
        self,
        file_stem: str,
        clusters: list[tuple[int, tuple[float, float, float], int]],
        logger: logging.Logger,
        is_first: bool = False,
        is_last: bool = False,
    ) -> dict[int, int]:
        """
        Update tracker with clusters from the current timestep.

        Parameters
        ----------
        file_stem:
            Filename stem of the current timestep.
        clusters:
            List of (label_id, cog_zyx, voxel_count).
            cog_zyx is a plain tuple — converted to np.ndarray once here,
            not repeatedly inside _match().
        is_first:
            Seeds all unmatched clusters unconditionally.
        is_last:
            Seeds any unmatched clusters not already tracked.

        Returns
        -------
        dict mapping label_id -> track_id for all matched/seeded clusters.
        """
        self._timestep_count += 1
        label_to_track: dict[int, int] = {}

        if not clusters:
            self._mark_all_lost(file_stem, logger)
            return label_to_track

        # Convert all incoming CoGs to np.ndarray once — not per distance call
        converted: list[tuple[int, np.ndarray, int]] = [
            (label_id, np.asarray(cog, dtype=np.float64), voxel_count)
            for label_id, cog, voxel_count in clusters
        ]

        matched_track_ids: set[int] = set()

        # Snapshot of tracks that existed BEFORE this timestep's loop begins.
        # At seeding timesteps (is_first/is_last), we only match against
        # pre-existing tracks — NOT against tracks seeded earlier in this
        # same loop. This prevents cluster N from matching a track that was
        # just seeded by cluster N-1 in the same timestep.
        pre_existing_track_ids: set[int] = set(self._tracks.keys())

        for label_id, cog_arr, voxel_count in converted:
            # Match only against tracks that existed before this timestep started
            track_id = self._match_within(cog_arr, pre_existing_track_ids)

            if track_id is not None:
                track = self._tracks[track_id]
                track.cog = cog_arr
                track.voxel_count = voxel_count
                track.last_seen = file_stem
                track.status = "active"
                track.timesteps_seen += 1
                track.timesteps_lost = 0
                matched_track_ids.add(track_id)
                label_to_track[label_id] = track_id
                logger.info(
                    "cluster %02d matched -> track %02d | voxels=%d | "
                    "cog=(%.1f, %.1f, %.1f)",
                    label_id, track_id, voxel_count,
                    cog_arr[0], cog_arr[1], cog_arr[2],
                )
            elif is_first or is_last:
                track_id = self._seed(file_stem, cog_arr, voxel_count)
                matched_track_ids.add(track_id)
                label_to_track[label_id] = track_id
                logger.info(
                    "cluster %02d seeded  -> track %02d | voxels=%d | "
                    "cog=(%.1f, %.1f, %.1f)",
                    label_id, track_id, voxel_count,
                    cog_arr[0], cog_arr[1], cog_arr[2],
                )

        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids and track.status == "active":
                track.status = "lost"
                track.timesteps_lost += 1
                logger.warning(
                    "track %02d LOST at timestep %s (last seen: %s | voxels: %d)",
                    track_id, file_stem, track.last_seen, track.voxel_count,
                )

        return label_to_track

    def is_tracked(self, label_id: int, label_to_track: dict[int, int]) -> bool:
        """Return True if label_id appears in the current timestep's track mapping."""
        return label_id in label_to_track

    def summary(self) -> list[dict]:
        """Return a list of dicts describing all tracks, suitable for a CSV or DuckDB insert."""
        return [
            {
                "track_id": t.track_id,
                "status": t.status,
                "first_seen": t.first_seen,
                "last_seen": t.last_seen,
                "timesteps_seen": t.timesteps_seen,
                "timesteps_lost": t.timesteps_lost,
                "final_voxel_count": t.voxel_count,
                "final_cog_z": float(t.cog[0]),
                "final_cog_y": float(t.cog[1]),
                "final_cog_x": float(t.cog[2]),
            }
            for t in sorted(self._tracks.values(), key=lambda x: x.track_id)
        ]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _match(self, cog_arr: np.ndarray) -> int | None:
        """Match against all active tracks (used during non-seeding timesteps)."""
        return self._match_within(cog_arr, set(self._tracks.keys()))

    def _match_within(
        self, cog_arr: np.ndarray, allowed_ids: set[int]
    ) -> int | None:
        """
        Find the nearest active track to cog_arr, restricted to allowed_ids.

        By passing the snapshot of pre-existing track IDs taken before the
        current timestep's loop, we prevent a cluster from matching a track
        that was just seeded by a previous cluster in the same loop iteration.

        cog_arr and track.cog are both np.ndarray — no temporary allocation.
        """
        best_id = None
        best_dist = self.max_cog_distance

        for track_id, track in self._tracks.items():
            if track_id not in allowed_ids:
                continue
            if track.status != "active":
                continue
            dist = float(np.linalg.norm(cog_arr - track.cog))
            if dist < best_dist:
                best_dist = dist
                best_id = track_id

        return best_id

    def _seed(
        self,
        file_stem: str,
        cog_arr: np.ndarray,
        voxel_count: int,
    ) -> int:
        """Seed a new track. cog_arr is already np.ndarray."""
        track_id = self._next_id
        self._next_id += 1
        self._tracks[track_id] = TrackedCluster(
            track_id=track_id,
            first_seen=file_stem,
            last_seen=file_stem,
            cog=cog_arr,
            voxel_count=voxel_count,
        )
        return track_id

    def _mark_all_lost(self, file_stem: str, logger: logging.Logger) -> None:
        for track_id, track in self._tracks.items():
            if track.status == "active":
                track.status = "lost"
                track.timesteps_lost += 1
                logger.warning(
                    "track %02d LOST at timestep %s (no clusters found)",
                    track_id, file_stem,
                )