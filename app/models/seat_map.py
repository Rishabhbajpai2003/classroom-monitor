"""
Seat calibration, seat map generation, and camera-motion-aware seat projection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RowGuide:
    row_id: str
    left: list[float]
    right: list[float]
    seat_count: int


@dataclass
class SeatCalibration:
    reference_frame_index: int
    frame_width: int
    frame_height: int
    front_edge: list[list[float]]
    rows: list[RowGuide]
    exit_polygon: list[list[float]] = field(default_factory=list)
    calibration_name: str = "seat_map_calibration"


@dataclass
class SeatDefinition:
    seat_id: str
    row_id: str
    seat_index: int
    row_rank: int
    seat_rank: int
    ref_point: list[float]
    front_distance: float
    match_radius: float


@dataclass
class SeatProjectionResult:
    frame_idx: int
    stable: bool
    homography: Optional[np.ndarray]
    seat_points: dict[str, Optional[list[float]]]
    seat_visibility: dict[str, str]
    exit_polygon: list[list[float]]
    inlier_count: int = 0


def _point_line_distance(point_xy: np.ndarray, line_a: np.ndarray, line_b: np.ndarray) -> float:
    line_vec = line_b - line_a
    denom = float(np.linalg.norm(line_vec))
    if denom <= 1e-6:
        return float(np.linalg.norm(point_xy - line_a))
    return float(abs(np.cross(line_vec, point_xy - line_a)) / denom)


def _normalize_point(point) -> list[float]:
    return [float(point[0]), float(point[1])]


def load_seat_calibration(path: str | Path) -> SeatCalibration:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = [
        RowGuide(
            row_id=str(row["row_id"]),
            left=[float(row["left"][0]), float(row["left"][1])],
            right=[float(row["right"][0]), float(row["right"][1])],
            seat_count=int(row["seat_count"]),
        )
        for row in payload.get("rows", [])
    ]
    return SeatCalibration(
        reference_frame_index=int(payload.get("reference_frame_index", 0)),
        frame_width=int(payload.get("frame_width", 0)),
        frame_height=int(payload.get("frame_height", 0)),
        front_edge=[
            _normalize_point(payload["front_edge"][0]),
            _normalize_point(payload["front_edge"][1]),
        ],
        rows=rows,
        exit_polygon=[_normalize_point(point) for point in payload.get("exit_polygon", [])],
        calibration_name=str(payload.get("calibration_name", "seat_map_calibration")),
    )


def save_seat_calibration(calibration: SeatCalibration, path: str | Path) -> str:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(calibration)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def build_seat_map(calibration: SeatCalibration) -> list[SeatDefinition]:
    if len(calibration.front_edge) != 2:
        raise ValueError("front_edge must contain exactly 2 points")
    if not calibration.rows:
        raise ValueError("At least one row guide is required")

    front_a = np.asarray(calibration.front_edge[0], dtype=np.float32)
    front_b = np.asarray(calibration.front_edge[1], dtype=np.float32)
    provisional: list[dict] = []

    for row in calibration.rows:
        if row.seat_count <= 0:
            continue
        left = np.asarray(row.left, dtype=np.float32)
        right = np.asarray(row.right, dtype=np.float32)
        seat_count = int(row.seat_count)
        step = 0.0 if seat_count == 1 else 1.0 / float(seat_count - 1)
        spacing = float(np.linalg.norm(right - left) / max(1, seat_count - 1)) if seat_count > 1 else 48.0
        row_mid = 0.5 * (left + right)
        row_front_distance = _point_line_distance(row_mid, front_a, front_b)
        for seat_idx in range(seat_count):
            alpha = 0.5 if seat_count == 1 else seat_idx * step
            point = (1.0 - alpha) * left + alpha * right
            provisional.append(
                {
                    "row_id": row.row_id,
                    "seat_index": seat_idx + 1,
                    "ref_point": _normalize_point(point),
                    "row_front_distance": row_front_distance,
                    "match_radius": max(30.0, spacing * 0.60),
                }
            )

    row_order = {
        row_id: rank + 1
        for rank, (row_id, _) in enumerate(
            sorted(
                {
                    row.row_id: _point_line_distance(
                        0.5 * (np.asarray(row.left, dtype=np.float32) + np.asarray(row.right, dtype=np.float32)),
                        front_a,
                        front_b,
                    )
                    for row in calibration.rows
                }.items(),
                key=lambda item: item[1],
            )
        )
    }

    sorted_seats = sorted(provisional, key=lambda item: (row_order[item["row_id"]], item["seat_index"]))
    seat_rank_lookup = {
        (seat["row_id"], seat["seat_index"]): rank + 1 for rank, seat in enumerate(sorted_seats)
    }

    seat_map: list[SeatDefinition] = []
    for seat in sorted_seats:
        row_rank = row_order[seat["row_id"]]
        seat_index = int(seat["seat_index"])
        seat_id = f"{seat['row_id']}-S{seat_index:02d}"
        seat_map.append(
            SeatDefinition(
                seat_id=seat_id,
                row_id=str(seat["row_id"]),
                seat_index=seat_index,
                row_rank=row_rank,
                seat_rank=seat_rank_lookup[(seat["row_id"], seat_index)],
                ref_point=list(seat["ref_point"]),
                front_distance=float(seat["row_front_distance"]),
                match_radius=float(seat["match_radius"]),
            )
        )
    return seat_map


def render_seat_map(
    reference_frame: np.ndarray,
    calibration: SeatCalibration,
    seat_map: list[SeatDefinition],
) -> np.ndarray:
    frame = reference_frame.copy()
    if len(calibration.front_edge) == 2:
        p1 = tuple(int(round(v)) for v in calibration.front_edge[0])
        p2 = tuple(int(round(v)) for v in calibration.front_edge[1])
        cv2.line(frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "FRONT", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    for row in calibration.rows:
        left = tuple(int(round(v)) for v in row.left)
        right = tuple(int(round(v)) for v in row.right)
        cv2.line(frame, left, right, (255, 180, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, row.row_id, left, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA)
    if len(calibration.exit_polygon) >= 3:
        polygon = np.asarray(calibration.exit_polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, "EXIT", tuple(polygon[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    for seat in seat_map:
        center = tuple(int(round(v)) for v in seat.ref_point)
        cv2.circle(frame, center, 5, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            seat.seat_id,
            (center[0] + 6, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def save_seat_map_json(
    seat_map: list[SeatDefinition],
    calibration: SeatCalibration,
    path: str | Path,
) -> str:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "calibration_name": calibration.calibration_name,
        "reference_frame_index": calibration.reference_frame_index,
        "frame_width": calibration.frame_width,
        "frame_height": calibration.frame_height,
        "front_edge": calibration.front_edge,
        "exit_polygon": calibration.exit_polygon,
        "seats": [asdict(seat) for seat in seat_map],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def save_seat_map_png(
    reference_frame: np.ndarray,
    calibration: SeatCalibration,
    seat_map: list[SeatDefinition],
    path: str | Path,
) -> str:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rendered = render_seat_map(reference_frame, calibration, seat_map)
    cv2.imwrite(path, rendered)
    return path


class CameraMotionCompensator:
    def __init__(
        self,
        reference_frame: np.ndarray,
        calibration: SeatCalibration,
        seat_map: list[SeatDefinition],
        min_matches: int = 16,
        min_inliers: int = 12,
    ):
        self.reference_frame = reference_frame
        self.calibration = calibration
        self.seat_map = seat_map
        self.min_matches = max(8, int(min_matches))
        self.min_inliers = max(6, int(min_inliers))
        self._orb = cv2.ORB_create(nfeatures=2500)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self._ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        self._ref_keypoints, self._ref_descriptors = self._orb.detectAndCompute(self._ref_gray, None)

        self._ref_seat_points = np.asarray([seat.ref_point for seat in seat_map], dtype=np.float32).reshape(-1, 1, 2)
        self._ref_exit_points = (
            np.asarray(calibration.exit_polygon, dtype=np.float32).reshape(-1, 1, 2)
            if calibration.exit_polygon
            else None
        )

    def project(self, frame: np.ndarray, frame_idx: int) -> SeatProjectionResult:
        if self._ref_descriptors is None or len(self._ref_descriptors) < self.min_matches:
            return SeatProjectionResult(
                frame_idx=frame_idx,
                stable=False,
                homography=None,
                seat_points={seat.seat_id: None for seat in self.seat_map},
                seat_visibility={seat.seat_id: "unstable_view" for seat in self.seat_map},
                exit_polygon=[],
                inlier_count=0,
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < self.min_matches:
            return SeatProjectionResult(
                frame_idx=frame_idx,
                stable=False,
                homography=None,
                seat_points={seat.seat_id: None for seat in self.seat_map},
                seat_visibility={seat.seat_id: "unstable_view" for seat in self.seat_map},
                exit_polygon=[],
                inlier_count=0,
            )

        matches = sorted(self._matcher.match(self._ref_descriptors, descriptors), key=lambda match: match.distance)
        if len(matches) < self.min_matches:
            return SeatProjectionResult(
                frame_idx=frame_idx,
                stable=False,
                homography=None,
                seat_points={seat.seat_id: None for seat in self.seat_map},
                seat_visibility={seat.seat_id: "unstable_view" for seat in self.seat_map},
                exit_polygon=[],
                inlier_count=len(matches),
            )

        ref_pts = np.float32([self._ref_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        cur_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(ref_pts, cur_pts, cv2.RANSAC, 5.0)
        inlier_count = int(mask.sum()) if mask is not None else 0
        if homography is None or inlier_count < self.min_inliers:
            return SeatProjectionResult(
                frame_idx=frame_idx,
                stable=False,
                homography=homography,
                seat_points={seat.seat_id: None for seat in self.seat_map},
                seat_visibility={seat.seat_id: "unstable_view" for seat in self.seat_map},
                exit_polygon=[],
                inlier_count=inlier_count,
            )

        projected = cv2.perspectiveTransform(self._ref_seat_points, homography).reshape(-1, 2)
        seat_points: dict[str, Optional[list[float]]] = {}
        seat_visibility: dict[str, str] = {}
        height, width = frame.shape[:2]
        margin = 8.0
        for seat, point in zip(self.seat_map, projected.tolist()):
            x, y = float(point[0]), float(point[1])
            if x < -margin or y < -margin or x > width + margin or y > height + margin:
                seat_points[seat.seat_id] = None
                seat_visibility[seat.seat_id] = "off_frame"
            else:
                seat_points[seat.seat_id] = [x, y]
                seat_visibility[seat.seat_id] = "visible"

        exit_polygon: list[list[float]] = []
        if self._ref_exit_points is not None and len(self._ref_exit_points) >= 3:
            projected_exit = cv2.perspectiveTransform(self._ref_exit_points, homography).reshape(-1, 2)
            exit_polygon = [[float(x), float(y)] for x, y in projected_exit.tolist()]

        return SeatProjectionResult(
            frame_idx=frame_idx,
            stable=True,
            homography=homography,
            seat_points=seat_points,
            seat_visibility=seat_visibility,
            exit_polygon=exit_polygon,
            inlier_count=inlier_count,
        )

