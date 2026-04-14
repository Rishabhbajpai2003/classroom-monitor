from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.models.detectors import OpenVocabularyObjectDetector, PersonDetector, bbox_iou
from app.models.pose_analyzer import PoseAnalyzer
from app.models.roster_policy import capped_reportable_ids, roster_size
from app.models.seat_events import SeatEventEngine
from app.models.seat_map import (
    CameraMotionCompensator,
    build_seat_map,
    load_seat_calibration,
    save_seat_map_json,
    save_seat_map_png,
)
from app.models.student_backbone import (
    SharedStudentBackbone,
    StudentObservation,
    format_student_display_name,
    seat_anchor_point,
    student_metadata_fields,
)
from app.utils.draw_utils import draw_labeled_box, draw_pose_skeleton
from app.utils.source_utils import is_youtube_url, resolve_source
from app.utils.vision_utils import normalize_list, point_to_box_distance
from detectors.activity_detector.config import NOTE_SURFACE_LIKE, PEN_LIKE

try:
    import mediapipe as mp
except ImportError:
    mp = None


PROMPT_ALIASES = {
    "paper": ["sheet of paper", "document", "handout", "notepad"],
    "sheet of paper": ["paper", "document", "handout"],
    "notebook": ["exercise notebook", "spiral notebook", "open notebook"],
    "worksheet": ["assignment sheet"],
    "book": ["textbook", "open book"],
    "writing": ["handwritten notes", "notes on paper"],
    "pen": ["ballpoint pen", "ink pen"],
    "pencil": ["wooden pencil", "mechanical pencil"],
    "cell phone": ["mobile phone", "smartphone", "phone"],
    "tablet": ["tablet computer", "ipad"],
    "laptop": ["laptop computer"],
}


@dataclass
class StudentActivityRecord:
    track_id: int
    seen_frames: int = 0
    counts: dict[str, int] = field(
        default_factory=lambda: {"note-taking": 0, "electronics": 0, "idle": 0, "unknown": 0}
    )
    ema_scores: dict[str, float] = field(
        default_factory=lambda: {"note-taking": 0.0, "electronics": 0.0, "idle": 0.0}
    )
    last_primary: str = "unknown"
    last_confidence: float = 0.0
    last_mode: str = "limited"
    last_reason: str = "no-observations"
    proof_links: dict[str, str] = field(default_factory=dict)
    proof_scores: dict[str, float] = field(default_factory=dict)


class ActivityIntervalTracker:
    def __init__(self, fps: float):
        self.fps = max(1e-6, float(fps))
        self.current: dict[str, dict] = {}
        self.intervals: list[dict] = []

    def update(
        self,
        student_id: str,
        track_id: int,
        frame_idx: int,
        activity_primary: str,
        confidence: float,
        mode: str,
        reason: str,
        seat_id: str = "",
        proof_keyframe: str = "",
    ):
        current = self.current.get(student_id)
        payload = {
            "track_id": track_id,
            "activity_primary": activity_primary,
            "activity_confidence": float(confidence),
            "activity_mode": mode,
            "activity_reason": reason,
            "seat_id": seat_id,
            "proof_keyframe": proof_keyframe,
        }
        if current is None:
            self.current[student_id] = {"start_frame": frame_idx, **payload}
            return

        same_activity = (
            current["activity_primary"] == activity_primary
            and current["activity_mode"] == mode
            and current["seat_id"] == seat_id
        )
        if same_activity:
            current["activity_confidence"] = max(float(current["activity_confidence"]), float(confidence))
            current["activity_reason"] = reason
            if proof_keyframe:
                current["proof_keyframe"] = proof_keyframe
            current["track_id"] = track_id
            return

        self._close_interval(student_id, frame_idx - 1)
        self.current[student_id] = {"start_frame": frame_idx, **payload}

    def _close_interval(self, student_id: str, end_frame: int):
        current = self.current.get(student_id)
        if current is None:
            return
        start_frame = int(current["start_frame"])
        end_frame = max(start_frame, int(end_frame))
        self.intervals.append(
            {
                "record_type": "interval",
                "person_id": student_id,
                "track_id": current["track_id"],
                "seat_id": current["seat_id"],
                "activity_primary": current["activity_primary"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_seconds": round((end_frame - start_frame + 1) / self.fps, 3),
                "activity_confidence": round(float(current["activity_confidence"]), 4),
                "activity_mode": current["activity_mode"],
                "activity_reason": current["activity_reason"],
                "proof_keyframe": current["proof_keyframe"],
            }
        )

    def finalize(self, total_frames: int):
        for student_id in list(self.current.keys()):
            self._close_interval(student_id, total_frames)
            self.current.pop(student_id, None)

def _expand_targets(targets: list[str]) -> list[str]:
    expanded: list[str] = []
    seen = set()
    for target in targets:
        token = str(target).strip().lower()
        if not token:
            continue
        if token not in seen:
            seen.add(token)
            expanded.append(token)
        for alias in PROMPT_ALIASES.get(token, []):
            alias = str(alias).strip().lower()
            if alias and alias not in seen:
                seen.add(alias)
                expanded.append(alias)
    return expanded


def _build_runtime_config(args) -> dict:
    device = args.device or "auto"
    return {
        "system": {
            "device": device,
            "output_dir": str(Path(args.out).resolve().parent),
            "camera_id": "cam_01",
            "frame_skip": 1,
            "save_video": True,
            "save_csv": True,
        },
        "detection": {
            "model_size": "yolo26s",
            "confidence_threshold": args.person_conf,
            "image_size": args.person_imgsz,
            "tile_grid": args.tile_grid,
            "tile_overlap": args.tile_overlap,
            "max_det": 350,
            "iou_threshold": 0.65,
        },
        "object_detection": {
            "weights": args.weights,
            "confidence_threshold": args.object_conf,
            "image_size": args.object_imgsz,
            "tile_grid": args.tile_grid,
            "tile_overlap": args.tile_overlap,
            "max_det": 500,
            "iou_threshold": 0.65,
        },
        "pose": {
            "weights": args.pose_weights,
            "confidence_threshold": args.pose_conf,
            "image_size": args.pose_imgsz,
            "iou_threshold": 0.85,
            "max_det": 300,
        },
        "student_backbone": {
            "face_process_fps": args.track_fps,
            "face_det_size": args.face_det_size,
            "face_det_thresh": 0.20,
            "face_tile_grid": args.face_tile_grid,
            "face_tile_overlap": args.face_tile_overlap,
            "face_min_size": args.min_face,
            "primary_detector": args.primary_detector,
            "backup_detector": args.backup_detector,
            "enable_backup_detector": not bool(args.disable_backup_detector),
            "adaface_weights": args.adaface_weights,
            "identity_db_path": args.identity_db,
            "identity_db_save_every": args.identity_db_save_every,
            "full_min_height": args.full_min_height,
            "reduced_min_height": args.reduced_min_height,
            "body_memory_frames": 18,
            "iou_weight": 0.22,
            "sim_weight": 0.70,
            "dist_weight": 0.08,
            "min_confirm_hits": 2,
            "reid_sim_thresh": 0.52,
            "appearance_weight": 0.16,
            "min_face_sim": 0.20,
            "merge_sim_thresh": 0.60,
            "young_track_hits": 10,
            "new_id_confirm_hits": 2,
            "new_id_confirm_quality": 0.20,
            "provisional_match_margin": 0.02,
            "high_det_score": 0.48,
            "continuity_window": 4,
            "continuity_iou_gate": 0.16,
            "continuity_dist_gate": 0.36,
            "continuity_relax": 0.12,
            "continuity_bonus": 0.14,
            "strong_named_match_score": 0.72,
        },
        "seating": {
            "calibration_path": args.seat_calibration,
            "initial_confirm_seconds": 1.0,
            "shift_confirm_seconds": 10.0,
            "seat_stick_seconds": 3.0,
        },
        "attendance_events": {
            "out_of_class_seconds": 20.0,
            "exit_zone_seconds": 8.0,
            "late_arrival_minutes": 5.0,
            "early_exit_minutes": 5.0,
        },
    }


def _safe_token(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    out = out.strip("_")
    return out or "activity"


def _write_dict_rows(csv_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _signal_rows(
    signal_name: str,
    records: dict[str, StudentActivityRecord],
    interval_rows: list[dict],
    seating_summary: dict[str, dict],
    student_metadata_map: dict[str, dict[str, str]],
    fps_out: float,
) -> list[dict]:
    rows: list[dict] = []
    for person_id in sorted(records.keys()):
        record = records[person_id]
        signal_frames = int(record.counts.get(signal_name, 0))
        if signal_frames <= 0 and record.last_primary != signal_name:
            continue
        seat_info = seating_summary.get(person_id, {})
        meta = dict(
            student_metadata_map.get(
                person_id,
                {"display_name": "Unknown", "student_name": "", "roll_number": "", "student_key": ""},
            )
        )
        rows.append(
            {
                "record_type": "summary",
                "person_id": person_id,
                "display_name": meta.get("display_name", ""),
                "student_name": meta.get("student_name", ""),
                "roll_number": meta.get("roll_number", ""),
                "student_key": meta.get("student_key", ""),
                "track_id": record.track_id,
                "seat_id": seat_info.get("seat_id") or "",
                "seat_rank": seat_info.get("seat_rank") or "",
                "row_rank": seat_info.get("row_rank") or "",
                "weighted_avg_seat_rank": seat_info.get("weighted_avg_seat_rank") or "",
                "signal_name": signal_name,
                "signal_frames": signal_frames,
                "signal_seconds": round(signal_frames / max(1e-6, fps_out), 3),
                "last_activity_primary": record.last_primary,
                "activity_confidence": round(record.last_confidence, 4),
                "activity_mode": record.last_mode,
                "activity_reason": record.last_reason,
                "proof_keyframe": record.proof_links.get(signal_name, ""),
            }
        )
    for row in interval_rows:
        if row.get("activity_primary") != signal_name:
            continue
        person_id = row["person_id"]
        seat_info = seating_summary.get(person_id, {})
        meta = dict(
            student_metadata_map.get(
                person_id,
                {"display_name": "Unknown", "student_name": "", "roll_number": "", "student_key": ""},
            )
        )
        rows.append(
            {
                "record_type": "interval",
                "person_id": person_id,
                "display_name": meta.get("display_name", ""),
                "student_name": meta.get("student_name", ""),
                "roll_number": meta.get("roll_number", ""),
                "student_key": meta.get("student_key", ""),
                "track_id": row.get("track_id", ""),
                "seat_id": row.get("seat_id", "") or seat_info.get("seat_id") or "",
                "seat_rank": seat_info.get("seat_rank") or "",
                "row_rank": seat_info.get("row_rank") or "",
                "weighted_avg_seat_rank": seat_info.get("weighted_avg_seat_rank") or "",
                "signal_name": signal_name,
                "signal_frames": "",
                "signal_seconds": row.get("duration_seconds", 0.0),
                "last_activity_primary": row.get("activity_primary", ""),
                "activity_confidence": row.get("activity_confidence", 0.0),
                "activity_mode": row.get("activity_mode", ""),
                "activity_reason": row.get("activity_reason", ""),
                "start_frame": row.get("start_frame", ""),
                "end_frame": row.get("end_frame", ""),
                "duration_seconds": row.get("duration_seconds", 0.0),
                "proof_keyframe": row.get("proof_keyframe", ""),
            }
        )
    return rows


def _reference_box(obs: StudentObservation) -> list[float]:
    return list(obs.body_bbox if obs.body_bbox is not None else obs.face_bbox)


def _box_center(box) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def _contact_score(a_box, b_box, max_dist: float) -> float:
    points = np.array(
        [
            [_box_center(a_box)[0], _box_center(a_box)[1]],
            [_box_center(b_box)[0], _box_center(b_box)[1]],
        ],
        dtype=np.float32,
    )
    dist = float(np.linalg.norm(points[0] - points[1]))
    return float(np.clip(1.0 - (dist / max(1e-6, max_dist)), 0.0, 1.0))


def _hand_points_from_pose(pose_result: dict) -> list[np.ndarray]:
    return [np.asarray(point, dtype=np.float32) for point in pose_result.get("hand_points", [])]


def _person_hand_near_box(hand_points: list[np.ndarray], box_xyxy, dist_thr: float) -> bool:
    if not hand_points:
        return False
    for hp in hand_points:
        if point_to_box_distance(hp, box_xyxy) <= dist_thr:
            return True
    return False


def _head_near_box(head_point: Optional[list[float]], box_xyxy, dist_thr: float) -> bool:
    if head_point is None:
        return False
    return point_to_box_distance(np.asarray(head_point, dtype=np.float32), box_xyxy) <= dist_thr


def _label_matches(name: str, keywords: set[str]) -> bool:
    token = str(name).lower().strip()
    return any((kw == token) or (kw in token) for kw in keywords)


def _init_hands(args):
    if not args.hands:
        return None
    if mp is None:
        raise RuntimeError("MediaPipe is not installed. Install with: pip install mediapipe==0.10.14")
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=args.max_hands,
        model_complexity=1,
        min_detection_confidence=args.hands_conf,
        min_tracking_confidence=args.hands_track,
    )


def _extract_roi_hands(frame: np.ndarray, body_box: list[float], hands) -> list[np.ndarray]:
    if hands is None or body_box is None:
        return []
    x1, y1, x2, y2 = [int(max(0, round(v))) for v in body_box]
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    points: list[np.ndarray] = []
    if res.multi_hand_landmarks:
        for hand_lms in res.multi_hand_landmarks:
            xs = [lm.x for lm in hand_lms.landmark]
            ys = [lm.y for lm in hand_lms.landmark]
            points.append(
                np.array(
                    [
                        x1 + float(np.mean(xs)) * max(1, x2 - x1),
                        y1 + float(np.mean(ys)) * max(1, y2 - y1),
                    ],
                    dtype=np.float32,
                )
            )
    return points


def _proof_threshold(args, mode: str) -> float:
    if mode == "full":
        return args.full_conf_threshold
    if mode == "reduced":
        return args.reduced_conf_threshold
    return args.limited_conf_threshold


def _draw_seat_overlay(frame: np.ndarray, seat_map, projection, seat_assignments: dict[str, str | None]) -> np.ndarray:
    occupied = {seat_id for seat_id in seat_assignments.values() if seat_id}
    for seat in seat_map:
        point = projection.seat_points.get(seat.seat_id)
        visibility = projection.seat_visibility.get(seat.seat_id, "unstable_view")
        if point is None:
            continue
        center = tuple(int(round(v)) for v in point)
        color = (0, 220, 0)
        if visibility == "off_frame":
            color = (120, 120, 120)
        elif visibility == "unstable_view":
            color = (0, 165, 255)
        elif seat.seat_id in occupied:
            color = (255, 180, 0)
        cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)
        cv2.putText(frame, seat.seat_id, (center[0] + 4, center[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    if projection.exit_polygon:
        polygon = np.asarray(projection.exit_polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [polygon], True, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def _update_activity_state(record: StudentActivityRecord, scores: dict[str, float], mode: str, reason: str):
    alpha = 0.35
    for label in record.ema_scores:
        record.ema_scores[label] = alpha * float(scores.get(label, 0.0)) + (1.0 - alpha) * record.ema_scores[label]

    ranked = sorted(record.ema_scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    threshold = 0.32 if mode == "full" else 0.40 if mode == "reduced" else 0.52
    if top_score < threshold:
        primary = "unknown"
        confidence = max(0.35, min(0.95, 1.0 - top_score))
    else:
        if record.last_primary in record.ema_scores and record.last_primary != top_label:
            previous_score = record.ema_scores[record.last_primary]
            if previous_score >= top_score - 0.06:
                top_label = record.last_primary
                top_score = previous_score
        primary = top_label
        confidence = max(0.30, min(0.98, top_score))

    record.last_primary = primary
    record.last_confidence = float(confidence)
    record.last_mode = mode
    record.last_reason = reason
    record.counts[primary] = record.counts.get(primary, 0) + 1
    return primary, float(confidence)


def _save_proof(
    proof_dir_path: Path,
    record: StudentActivityRecord,
    activity: str,
    confidence: float,
    raw_frame: np.ndarray,
    ref_box: list[float],
    frame_idx: int,
    object_box=None,
    object_label: Optional[str] = None,
):
    if confidence <= record.proof_scores.get(activity, -1.0):
        return

    proof_frame = raw_frame.copy()
    draw_labeled_box(proof_frame, ref_box, activity, (0, 255, 0), 2)
    if object_box is not None:
        draw_labeled_box(proof_frame, object_box, object_label or "object", (0, 255, 255), 2)

    frame_dir = proof_dir_path / f"frame_{int(frame_idx):06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{activity}_p{_safe_token(record.last_mode)}_{_safe_token(record.track_id)}.jpg"
    path = frame_dir / filename
    cv2.imwrite(str(path), proof_frame)
    record.proof_scores[activity] = float(confidence)
    record.proof_links[activity] = str(path.resolve())


def _classify_activity(
    args,
    obs: StudentObservation,
    pose_result: dict,
    relevant_objects: list[dict],
    hand_points: list[np.ndarray],
):
    mode = obs.size_mode
    ref_box = _reference_box(obs)
    ref_center = _box_center(ref_box)
    ref_size = max(1.0, max(ref_box[2] - ref_box[0], ref_box[3] - ref_box[1]))
    head_point = pose_result.get("head_point")
    head_forward = bool(pose_result.get("head_forward", False))

    phone_keywords = {"phone", "mobile phone", "smartphone", "cell phone"}
    laptop_keywords = {"laptop"}
    tablet_keywords = {"tablet", "ipad", "tab"}
    pen_keywords = set(PEN_LIKE)
    note_surface_keywords = set(NOTE_SURFACE_LIKE)

    phones = [obj for obj in relevant_objects if _label_matches(obj["class_name"], phone_keywords)]
    laptops = [obj for obj in relevant_objects if _label_matches(obj["class_name"], laptop_keywords)]
    tablets = [obj for obj in relevant_objects if _label_matches(obj["class_name"], tablet_keywords)]
    pens = [obj for obj in relevant_objects if _label_matches(obj["class_name"], pen_keywords)]
    note_surfaces = [
        obj
        for obj in relevant_objects
        if _label_matches(obj["class_name"], note_surface_keywords)
        and not _label_matches(obj["class_name"], pen_keywords)
        and not _label_matches(obj["class_name"], phone_keywords | laptop_keywords | tablet_keywords)
    ]

    def best_object_score(candidates, *, near_face_weight: float, near_hand_weight: float, near_body_weight: float):
        best = (0.0, None)
        for obj in candidates:
            bbox = obj["bbox"]
            obj_center = _box_center(bbox)
            near_body = max(0.0, 1.0 - np.linalg.norm(ref_center - obj_center) / max(1.0, ref_size))
            near_face = 0.0
            if head_point is not None:
                near_face = max(0.0, 1.0 - point_to_box_distance(np.asarray(head_point, dtype=np.float32), bbox) / max(1.0, 0.8 * ref_size))
            near_hand = 1.0 if _person_hand_near_box(hand_points, bbox, max(22.0, 0.18 * ref_size)) else 0.0
            score = (
                0.35 * float(obj.get("confidence", 0.0))
                + near_face_weight * near_face
                + near_hand_weight * near_hand
                + near_body_weight * near_body
            )
            if score > best[0]:
                best = (float(score), obj)
        return best

    phone_score, phone_obj = best_object_score(phones, near_face_weight=0.30, near_hand_weight=0.25, near_body_weight=0.10)
    laptop_score, laptop_obj = best_object_score(laptops, near_face_weight=0.12, near_hand_weight=0.16, near_body_weight=0.22)
    tablet_score, tablet_obj = best_object_score(tablets, near_face_weight=0.12, near_hand_weight=0.18, near_body_weight=0.18)
    pen_score, pen_obj = best_object_score(pens, near_face_weight=0.0, near_hand_weight=0.34, near_body_weight=0.08)
    surface_score, surface_obj = best_object_score(note_surfaces, near_face_weight=0.10, near_hand_weight=0.22, near_body_weight=0.18)

    electronics = max(
        phone_score,
        laptop_score + (0.10 if head_forward else 0.0),
        tablet_score + (0.06 if mode == "full" else 0.0),
    )
    note_taking = max(
        surface_score + 0.25 * pen_score + (0.10 if head_forward else 0.0),
        tablet_score + 0.30 * pen_score if tablet_obj is not None else 0.0,
    )

    reason = "weak-evidence"
    chosen_obj = None
    if phone_obj is not None and electronics == phone_score:
        reason = "phone-object"
        chosen_obj = phone_obj
    elif laptop_obj is not None and electronics >= laptop_score:
        reason = "laptop-object"
        chosen_obj = laptop_obj
    elif tablet_obj is not None and electronics >= tablet_score and note_taking < electronics:
        reason = "tablet-electronics"
        chosen_obj = tablet_obj
    if surface_obj is not None and note_taking >= electronics:
        reason = "note-surface"
        chosen_obj = surface_obj
    if tablet_obj is not None and note_taking > electronics and reason == "weak-evidence":
        reason = "tablet-note"
        chosen_obj = tablet_obj

    idle = 0.0
    if mode != "limited":
        pose_conf = float(pose_result.get("pose_confidence", 0.0))
        if head_forward and pose_conf >= 0.22 and electronics < max(args.phone_conf, args.laptop_conf) and note_taking < args.note_conf:
            idle = 0.38 + 0.28 * pose_conf
            reason = "head-forward-idle"

    if mode == "limited":
        if electronics >= args.phone_conf:
            scores = {"electronics": electronics, "note-taking": 0.0, "idle": 0.0}
            return scores, mode, reason, chosen_obj
        return {"electronics": 0.0, "note-taking": 0.0, "idle": 0.0}, mode, "presence-only", None

    if mode == "reduced":
        if note_taking < args.note_conf and electronics < min(args.phone_conf, args.laptop_conf, args.tablet_conf):
            idle = max(idle, 0.0)

    scores = {"electronics": electronics, "note-taking": note_taking, "idle": idle}
    return scores, mode, reason, chosen_obj


def run_pipeline(args):
    activity_csv_path = Path(args.activity_out)
    proof_dir_path = Path(args.proof_dir)
    if not proof_dir_path.is_absolute():
        proof_dir_path = proof_dir_path if len(proof_dir_path.parts) > 1 else activity_csv_path.parent / proof_dir_path
    if proof_dir_path.exists():
        shutil.rmtree(proof_dir_path)
    proof_dir_path.mkdir(parents=True, exist_ok=True)

    download_dir_path = Path(args.download_dir)
    if not download_dir_path.is_absolute():
        download_dir_path = activity_csv_path.parent / download_dir_path
    resolved_source, cleanup_source_path = resolve_source(
        args.source,
        download_dir=download_dir_path,
        keep_downloaded_source=args.keep_downloaded_source,
    )
    if is_youtube_url(args.source):
        print(f"Resolved YouTube source to local file: {resolved_source}")

    electronics_targets = _expand_targets(normalize_list(args.electronics))
    note_targets = _expand_targets(normalize_list(args.notetaking))
    all_object_targets = []
    seen = set()
    for name in electronics_targets + note_targets:
        if name not in seen:
            seen.add(name)
            all_object_targets.append(name)

    config = _build_runtime_config(args)
    person_detector = PersonDetector(config)
    object_detector = OpenVocabularyObjectDetector(all_object_targets, config=config, config_section="object_detection")
    pose_analyzer = PoseAnalyzer(config)
    student_backbone = SharedStudentBackbone(config)
    hands = _init_hands(args)

    cap = cv2.VideoCapture(resolved_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {resolved_source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out = fps_in if args.fps_out <= 0 else float(args.fps_out)

    out = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out,
        (width, height),
    )
    if not out.isOpened():
        raise RuntimeError(f"Could not open writer for {args.out}")

    track_every = max(1, int(round(fps_in / max(1e-6, args.track_fps))))
    object_every = max(1, int(round(fps_in / max(1e-6, args.object_fps))))
    pose_every = max(1, int(round(fps_in / max(1e-6, args.pose_fps))))

    cached_body_detections: list[dict] = []
    cached_objects: list[dict] = []
    cached_pose: dict[int, tuple[int, dict]] = {}
    records: dict[str, StudentActivityRecord] = {}
    student_metadata_map: dict[str, dict[str, str]] = {}
    interval_tracker = ActivityIntervalTracker(fps_out)
    overlay_min_unnamed_frames = max(24, int(round(fps_out * 0.75)))
    overlay_min_unnamed_seconds = 3.0
    roster_limit = roster_size("student-details")

    def remember_student_metadata(obs: StudentObservation) -> None:
        metadata = getattr(obs.face_track, "metadata", None) if obs.face_track is not None else None
        fields = student_metadata_fields(metadata)
        fields["display_name"] = format_student_display_name(obs.track_id, metadata)
        student_metadata_map[obs.global_id] = fields

    def student_metadata_row(student_id: str) -> dict[str, str]:
        return dict(
            student_metadata_map.get(
                student_id,
                {
                    "student_name": "",
                    "roll_number": "",
                    "student_key": "",
                    "display_name": "Unknown" if str(student_id).startswith("TEMP_") else str(student_id),
                },
            )
        )

    def should_render_overlay(student_id: str, seat_id: str = "") -> bool:
        if not str(student_id).startswith("TEMP_"):
            return True
        if roster_limit > 0:
            named_present = sum(1 for key in records if key and not str(key).startswith("TEMP_"))
            if named_present >= roster_limit:
                return False
        record = records.get(student_id)
        if record is None:
            return False
        return (
            record.seen_frames >= overlay_min_unnamed_frames
            and (record.seen_frames / max(1e-6, fps_out)) >= overlay_min_unnamed_seconds
            and bool(seat_id)
        )
    seat_event_engine = None
    seat_projection_manager = None
    seat_map = []
    seat_reference_frame = None
    seat_calibration = None
    if args.seat_calibration:
        seat_calibration = load_seat_calibration(args.seat_calibration)
        seat_map = build_seat_map(seat_calibration)
        ref_cap = cv2.VideoCapture(resolved_source)
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(seat_calibration.reference_frame_index)))
        ok_ref, reference_frame = ref_cap.read()
        ref_cap.release()
        if not ok_ref:
            raise RuntimeError("Could not read seat calibration reference frame.")
        seat_reference_frame = reference_frame
        seat_projection_manager = CameraMotionCompensator(reference_frame, seat_calibration, seat_map)
        seat_event_engine = SeatEventEngine(
            seat_map,
            fps=fps_in,
            initial_confirm_seconds=1.0,
            shift_confirm_seconds=10.0,
            seat_stick_seconds=3.0,
            out_of_class_seconds=20.0,
            exit_zone_seconds=8.0,
            late_arrival_minutes=5.0,
            early_exit_minutes=5.0,
        )

    frame_idx = 0
    written = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            raw_frame = frame.copy()
            if frame_idx % track_every == 0 or not cached_body_detections:
                cached_body_detections = person_detector.detect(frame)
            observations = student_backbone.step(frame, frame_idx, fps_in, body_detections=cached_body_detections)

            if frame_idx % object_every == 0 or not cached_objects:
                cached_objects = object_detector.detect(frame)

            pose_targets = [
                {"track_id": obs.track_id, "bbox": obs.body_bbox}
                for obs in observations
                if obs.body_bbox is not None and obs.size_mode != "limited"
            ]
            if pose_targets and frame_idx % pose_every == 0:
                pose_results = pose_analyzer.analyze_batch(frame, pose_targets)
                for track_id, pose_result in pose_results.items():
                    cached_pose[int(track_id)] = (frame_idx, pose_result)

            seat_assignments: dict[str, str | None] = {}
            seat_states: dict[str, str] = {}
            seat_projection = None
            if seat_projection_manager is not None and seat_event_engine is not None:
                seat_projection = seat_projection_manager.project(frame, frame_idx)
                seat_students = [
                    {
                        "global_id": obs.global_id,
                        "track_id": obs.track_id,
                        "center": seat_anchor_point(obs.body_bbox, obs.face_bbox).tolist(),
                    }
                    for obs in observations
                ]
                seat_assignments = seat_event_engine.update(frame_idx, seat_projection, seat_students)
                seat_states = {
                    obs.global_id: seat_event_engine.get_current_state(obs.global_id)
                    for obs in observations
                }

            for obj in cached_objects:
                color = (0, 255, 255)
                if _label_matches(obj["class_name"], set(PEN_LIKE)):
                    color = (0, 165, 255)
                elif _label_matches(obj["class_name"], set(NOTE_SURFACE_LIKE)):
                    color = (255, 0, 255)
                draw_labeled_box(frame, obj["bbox"], f"{obj['class_name']} {obj['confidence']:.2f}", color, 1)

            for obs in observations:
                remember_student_metadata(obs)
                record = records.setdefault(obs.global_id, StudentActivityRecord(track_id=obs.track_id))
                record.track_id = obs.track_id
                record.seen_frames += 1
                seat_id = seat_assignments.get(obs.global_id) or ""
                seat_state = seat_states.get(obs.global_id, "unassigned")
                pose_result = pose_analyzer._default_result()
                cached = cached_pose.get(obs.track_id)
                if cached is not None and frame_idx - int(cached[0]) <= max(2, pose_every * 2):
                    pose_result = cached[1]

                ref_box = _reference_box(obs)
                ref_size = max(1.0, max(ref_box[2] - ref_box[0], ref_box[3] - ref_box[1]))
                relevant_objects = []
                for obj in cached_objects:
                    if bbox_iou(ref_box, obj["bbox"]) > 0.02:
                        relevant_objects.append(obj)
                        continue
                    if np.linalg.norm(_box_center(ref_box) - _box_center(obj["bbox"])) <= 0.85 * ref_size:
                        relevant_objects.append(obj)

                hand_points = _hand_points_from_pose(pose_result)
                if (
                    hands is not None
                    and obs.body_bbox is not None
                    and obs.size_mode == "full"
                    and relevant_objects
                ):
                    hand_points.extend(_extract_roi_hands(frame, obs.body_bbox, hands))

                scores, mode, reason, chosen_obj = _classify_activity(
                    args,
                    obs,
                    pose_result,
                    relevant_objects,
                    hand_points,
                )
                primary, confidence = _update_activity_state(record, scores, mode, reason)
                proof_link = record.proof_links.get(primary, "")

                color_map = {
                    "note-taking": (255, 0, 255),
                    "electronics": (0, 255, 0),
                    "idle": (255, 180, 0),
                    "unknown": (140, 140, 140),
                }
                render_overlay = should_render_overlay(obs.global_id, seat_id=seat_id)
                if render_overlay:
                    display_name = student_metadata_row(obs.global_id).get("display_name", obs.global_id)
                    label_parts = [display_name]
                    if seat_id:
                        label_parts.append(seat_id)
                    label_parts.append(primary)
                    label_parts.append(f"{confidence:.2f}")
                    draw_labeled_box(
                        frame,
                        ref_box,
                        " | ".join(label_parts),
                        color_map.get(primary, (255, 255, 255)),
                        2,
                    )
                if (
                    render_overlay
                    and pose_result.get("keypoints_xy")
                    and (not str(obs.global_id).startswith("TEMP_") or confidence >= 0.55)
                ):
                    draw_pose_skeleton(
                        frame,
                        np.asarray(pose_result.get("keypoints_xy"), dtype=np.float32),
                        np.asarray(pose_result.get("keypoints_conf"), dtype=np.float32),
                        conf_thr=0.35,
                    )

                if primary != "unknown" and confidence >= _proof_threshold(args, mode):
                    _save_proof(
                        proof_dir_path,
                        record,
                        primary,
                        confidence,
                        raw_frame,
                        ref_box,
                        frame_idx,
                        object_box=None if chosen_obj is None else chosen_obj["bbox"],
                        object_label=None if chosen_obj is None else chosen_obj["class_name"],
                    )
                    proof_link = record.proof_links.get(primary, "")

                interval_tracker.update(
                    obs.global_id,
                    obs.track_id,
                    frame_idx,
                    primary,
                    confidence,
                    mode,
                    reason,
                    seat_id=seat_id,
                    proof_keyframe=proof_link,
                )

            if seat_projection is not None:
                frame = _draw_seat_overlay(frame, seat_map, seat_projection, seat_assignments)

            out.write(frame)
            written += 1
            frame_idx += 1
    finally:
        cap.release()
        out.release()
        student_backbone.close()
        if hands is not None:
            hands.close()

    final_frame_idx = max(0, frame_idx - 1)
    interval_tracker.finalize(final_frame_idx)
    if seat_event_engine is not None:
        seat_event_engine.finalize(final_frame_idx)

    seating_summary = seat_event_engine.get_student_summary() if seat_event_engine is not None else {}
    report_min_frames = max(60, int(round(fps_out * 2.0)))
    report_min_seconds = 8.0
    named_ids: list[str] = []
    unnamed_candidates: list[tuple[str, float]] = []
    for person_id, record in records.items():
        person_key = str(person_id)
        if not person_key:
            continue
        if not person_key.startswith("TEMP_"):
            named_ids.append(person_key)
            continue
        if (
            record.seen_frames >= report_min_frames
            and (record.seen_frames / max(1e-6, fps_out)) >= report_min_seconds
        ):
            unnamed_candidates.append((person_key, float(record.seen_frames) + float(max(record.proof_scores.values(), default=0.0))))
    roster_limit = roster_size("student-details")
    if roster_limit > 0:
        reportable_person_ids = capped_reportable_ids(named_ids, unnamed_candidates, roster_limit=roster_limit)
    else:
        reportable_person_ids = {item for item in named_ids if item}
        reportable_person_ids.update(item for item, _score in unnamed_candidates if item)

    if seat_calibration is not None and seat_reference_frame is not None:
        seat_map_json_path = activity_csv_path.parent / "seat_map.json"
        seat_map_png_path = activity_csv_path.parent / "seat_map.png"
        save_seat_map_json(seat_map, seat_calibration, seat_map_json_path)
        save_seat_map_png(seat_reference_frame, seat_calibration, seat_map, seat_map_png_path)
        if seat_event_engine is not None:
            timeline_path = activity_csv_path.parent / "student_seating_timeline.csv"
            event_path = activity_csv_path.parent / "attendance_events.csv"
            timeline_rows = [
                row
                for row in seat_event_engine.get_timeline_rows()
                if (
                    row.get("student_id", "") in reportable_person_ids
                    and row.get("state") == "seated"
                    and row.get("seat_id")
                    and row.get("seat_id") != "unassigned"
                )
            ]
            with open(timeline_path, "w", newline="", encoding="utf-8") as handle:
                timeline_writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "student_id",
                        "track_id",
                        "seat_id",
                        "seat_rank",
                        "row_rank",
                        "state",
                        "start_frame",
                        "end_frame",
                        "duration_seconds",
                    ],
                )
                timeline_writer.writeheader()
                timeline_writer.writerows(timeline_rows)
            with open(event_path, "w", newline="", encoding="utf-8") as handle:
                event_rows = [
                    row
                    for row in seat_event_engine.get_event_rows()
                    if row.get("student_id", "") in reportable_person_ids
                ]
                fieldnames = sorted({key for row in event_rows for key in row.keys()}) if event_rows else [
                    "student_id",
                    "track_id",
                    "event_type",
                    "seat_id",
                    "from_seat",
                    "to_seat",
                    "start_frame",
                    "end_frame",
                    "duration_seconds",
                    "reason",
                    "confidence",
                ]
                event_writer = csv.DictWriter(handle, fieldnames=fieldnames)
                event_writer.writeheader()
                event_writer.writerows(event_rows)

    interval_rows = interval_tracker.intervals
    with open(activity_csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "record_type",
            "person_id",
            "display_name",
            "student_name",
            "roll_number",
            "student_key",
            "track_id",
            "seat_id",
            "seat_rank",
            "row_rank",
            "weighted_avg_seat_rank",
            "seat_state",
            "seen_frames",
            "seen_seconds",
            "dominant_activity",
            "dominant_frames",
            "dominant_seconds",
            "activity_primary",
            "activity_confidence",
            "activity_mode",
            "activity_reason",
            "electronics_frames",
            "note_taking_frames",
            "idle_frames",
            "unknown_frames",
            "out_of_class_seconds",
            "late_arrival",
            "early_exit",
            "start_frame",
            "end_frame",
            "duration_seconds",
            "proof_keyframe",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for person_id in sorted(records.keys()):
            if person_id not in reportable_person_ids:
                continue
            record = records[person_id]
            dominant_activity, dominant_frames = max(record.counts.items(), key=lambda item: item[1])
            proof_link = record.proof_links.get(dominant_activity, "")
            seat_info = seating_summary.get(person_id, {})
            meta = student_metadata_row(person_id)
            writer.writerow(
                {
                    "record_type": "summary",
                    "person_id": person_id,
                    "display_name": meta.get("display_name", ""),
                    "student_name": meta.get("student_name", ""),
                    "roll_number": meta.get("roll_number", ""),
                    "student_key": meta.get("student_key", ""),
                    "track_id": record.track_id,
                    "seat_id": seat_info.get("seat_id") or "",
                    "seat_rank": seat_info.get("seat_rank") or "",
                    "row_rank": seat_info.get("row_rank") or "",
                    "weighted_avg_seat_rank": seat_info.get("weighted_avg_seat_rank") or "",
                    "seat_state": seat_info.get("current_state") or "unassigned",
                    "seen_frames": record.seen_frames,
                    "seen_seconds": round(record.seen_frames / max(1e-6, fps_out), 3),
                    "dominant_activity": dominant_activity,
                    "dominant_frames": dominant_frames,
                    "dominant_seconds": round(dominant_frames / max(1e-6, fps_out), 3),
                    "activity_primary": record.last_primary,
                    "activity_confidence": round(record.last_confidence, 4),
                    "activity_mode": record.last_mode,
                    "activity_reason": record.last_reason,
                    "electronics_frames": record.counts.get("electronics", 0),
                    "note_taking_frames": record.counts.get("note-taking", 0),
                    "idle_frames": record.counts.get("idle", 0),
                    "unknown_frames": record.counts.get("unknown", 0),
                    "out_of_class_seconds": seat_info.get("out_of_class_seconds") or 0.0,
                    "late_arrival": bool(seat_info.get("late_arrival")),
                    "early_exit": bool(seat_info.get("early_exit")),
                    "proof_keyframe": proof_link,
                }
            )
        for row in interval_rows:
            if row["person_id"] not in reportable_person_ids:
                continue
            seat_info = seating_summary.get(row["person_id"], {})
            meta = student_metadata_row(row["person_id"])
            writer.writerow(
                {
                    "record_type": row["record_type"],
                    "person_id": row["person_id"],
                    "display_name": meta.get("display_name", ""),
                    "student_name": meta.get("student_name", ""),
                    "roll_number": meta.get("roll_number", ""),
                    "student_key": meta.get("student_key", ""),
                    "track_id": row["track_id"],
                    "seat_id": row.get("seat_id", ""),
                    "seat_rank": seat_info.get("seat_rank") or "",
                    "row_rank": seat_info.get("row_rank") or "",
                    "weighted_avg_seat_rank": seat_info.get("weighted_avg_seat_rank") or "",
                    "seat_state": seat_info.get("current_state") or "unassigned",
                    "activity_primary": row["activity_primary"],
                    "activity_confidence": row["activity_confidence"],
                    "activity_mode": row["activity_mode"],
                    "activity_reason": row["activity_reason"],
                    "out_of_class_seconds": seat_info.get("out_of_class_seconds") or 0.0,
                    "late_arrival": bool(seat_info.get("late_arrival")),
                    "early_exit": bool(seat_info.get("early_exit")),
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "duration_seconds": row["duration_seconds"],
                    "proof_keyframe": row["proof_keyframe"],
                }
            )

    device_rows = [
        row
        for row in _signal_rows("electronics", records, interval_rows, seating_summary, student_metadata_map, fps_out)
        if row.get("person_id", "") in reportable_person_ids
    ]
    _write_dict_rows(
        activity_csv_path.parent / "device_use.csv",
        [
            "record_type",
            "person_id",
            "display_name",
            "student_name",
            "roll_number",
            "student_key",
            "track_id",
            "seat_id",
            "seat_rank",
            "row_rank",
            "weighted_avg_seat_rank",
            "signal_name",
            "signal_frames",
            "signal_seconds",
            "last_activity_primary",
            "activity_confidence",
            "activity_mode",
            "activity_reason",
            "start_frame",
            "end_frame",
            "duration_seconds",
            "proof_keyframe",
        ],
        device_rows,
    )
    note_rows = [
        row
        for row in _signal_rows("note-taking", records, interval_rows, seating_summary, student_metadata_map, fps_out)
        if row.get("person_id", "") in reportable_person_ids
    ]
    _write_dict_rows(
        activity_csv_path.parent / "note_taking.csv",
        [
            "record_type",
            "person_id",
            "display_name",
            "student_name",
            "roll_number",
            "student_key",
            "track_id",
            "seat_id",
            "seat_rank",
            "row_rank",
            "weighted_avg_seat_rank",
            "signal_name",
            "signal_frames",
            "signal_seconds",
            "last_activity_primary",
            "activity_confidence",
            "activity_mode",
            "activity_reason",
            "start_frame",
            "end_frame",
            "duration_seconds",
            "proof_keyframe",
        ],
        note_rows,
    )

    print(f"Done. Wrote {written} frames at {fps_out:.2f} FPS. Output: {args.out}")
    print(f"Activity summary saved to: {activity_csv_path}")
    print(f"Proof keyframes saved to: {proof_dir_path}")
    if cleanup_source_path is not None:
        cleanup_path = Path(cleanup_source_path)
        if cleanup_path.exists():
            cleanup_path.unlink()
            print(f"Deleted temporary downloaded source: {cleanup_path}")
