from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackedCluster:
    track_id: int
    first_seen: str
    last_seen: str
    cog: np.ndarray          # shape (3,) float64 — (z, y, x) voxel coords
    voxel_count: int
    status: str = "active"   # "active" | "lost"
    timesteps_seen: int = 1
    timesteps_lost: int = 0


@dataclass
class ClusterTracker:
    """Tracks clusters across timesteps by nearest centre-of-gravity match.

    Top-N from the first and last processed timesteps are always seeded; other
    timesteps only match against pre-existing tracks. Unmatched active tracks
    are marked "lost".
    """
    n_keep: int
    max_cog_distance: float = 1000.0
    _tracks: dict[int, TrackedCluster] = field(default_factory=dict)
    _next_id: int = field(default=1)
    _timestep_count: int = field(default=0)

    def update(
        self,
        file_stem: str,
        clusters: list[tuple[int, tuple[float, float, float], int]],
        logger: logging.Logger,
        is_first: bool = False,
        is_last: bool = False,
    ) -> dict[int, int]:
        """clusters: list of (label_id, cog_zyx, voxel_count). Returns label_id -> track_id."""
        self._timestep_count += 1
        label_to_track: dict[int, int] = {}

        if not clusters:
            self._mark_all_lost(file_stem, logger)
            return label_to_track

        converted = [
            (label_id, np.asarray(cog, dtype=np.float64), voxel_count)
            for label_id, cog, voxel_count in clusters
        ]

        matched_track_ids: set[int] = set()
        # Match only against tracks that existed before this timestep's loop, so a
        # cluster can't match a track just seeded by an earlier cluster this step.
        pre_existing_track_ids: set[int] = set(self._tracks.keys())

        for label_id, cog_arr, voxel_count in converted:
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
                logger.info("cluster %02d matched -> track %02d | voxels=%d | cog=(%.1f, %.1f, %.1f)",
                            label_id, track_id, voxel_count, cog_arr[0], cog_arr[1], cog_arr[2])
            elif is_first or is_last:
                track_id = self._seed(file_stem, cog_arr, voxel_count)
                matched_track_ids.add(track_id)
                label_to_track[label_id] = track_id
                logger.info("cluster %02d seeded  -> track %02d | voxels=%d | cog=(%.1f, %.1f, %.1f)",
                            label_id, track_id, voxel_count, cog_arr[0], cog_arr[1], cog_arr[2])

        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids and track.status == "active":
                track.status = "lost"
                track.timesteps_lost += 1
                logger.warning("track %02d LOST at timestep %s (last seen: %s | voxels: %d)",
                               track_id, file_stem, track.last_seen, track.voxel_count)

        return label_to_track

    def summary(self) -> list[dict]:
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

    def _match_within(self, cog_arr: np.ndarray, allowed_ids: set[int]) -> int | None:
        best_id = None
        best_dist = self.max_cog_distance
        for track_id, track in self._tracks.items():
            if track_id not in allowed_ids or track.status != "active":
                continue
            dist = float(np.linalg.norm(cog_arr - track.cog))
            if dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id

    def _seed(self, file_stem: str, cog_arr: np.ndarray, voxel_count: int) -> int:
        track_id = self._next_id
        self._next_id += 1
        self._tracks[track_id] = TrackedCluster(
            track_id=track_id, first_seen=file_stem, last_seen=file_stem,
            cog=cog_arr, voxel_count=voxel_count,
        )
        return track_id

    def _mark_all_lost(self, file_stem: str, logger: logging.Logger) -> None:
        for track_id, track in self._tracks.items():
            if track.status == "active":
                track.status = "lost"
                track.timesteps_lost += 1
                logger.warning("track %02d LOST at timestep %s (no clusters found)", track_id, file_stem)
