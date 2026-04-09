"""
Recall-first classroom attention, attendance, and seat-aware event processor.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

from app.models.attendance_manager import AttendanceManager
from app.models.detectors import OpenVocabularyObjectDetector, PersonDetector, bbox_iou
from app.models.hand_raise_events import HandRaiseEventTracker
from app.models.pose_analyzer import PoseAnalyzer
from app.models.seat_events import SeatEventEngine
from app.models.seat_map import (
    CameraMotionCompensator,
    build_seat_map,
    load_seat_calibration,
    save_seat_map_json,
    save_seat_map_png,
)
from app.models.student_backbone import SharedStudentBackbone, StudentObservation
from detectors.attention_detector.attention_engine import AttentionEngine

logger = logging.getLogger(__name__)


def _box_center(box: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


class ClassroomProcessor:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        sys_cfg = self.config.get("system", {})
        att_cfg = self.config.get("attention", {})
        seating_cfg = self.config.get("seating", {})
        event_cfg = self.config.get("attendance_events", {})

        self.camera_id = sys_cfg.get("camera_id", "cam_01")
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.camera_id}"
        self.output_dir = sys_cfg.get("output_dir", "outputs")
        self.save_video = bool(sys_cfg.get("save_video", True))
        self.save_csv = bool(sys_cfg.get("save_csv", True))
        self.headless = bool(sys_cfg.get("headless", False))
        self.frame_skip = int(sys_cfg.get("frame_skip", 1))

        self.phone_object_fps = float(att_cfg.get("phone_object_fps", 2.0))
        self.pose_fps = float(att_cfg.get("pose_fps", 3.0))

        self.person_detector = PersonDetector(self.config)
        self.phone_detector = OpenVocabularyObjectDetector(
            target_classes=["cell phone", "mobile phone", "smartphone", "phone"],
            config=self.config,
            config_section="object_detection",
        )
        self.pose_analyzer = PoseAnalyzer(self.config)
        self.student_backbone = SharedStudentBackbone(self.config)
        self.attendance_manager = AttendanceManager(self.config)
        self.attention_engine = AttentionEngine(self.config)

        self.seat_calibration_path = seating_cfg.get("calibration_path")
        self._seat_calibration = None
        self._seat_map = []
        self._projection_manager = None
        self._seat_event_engine = None
        self._hand_raise_tracker = None
        self._reference_frame = None

        self.initial_confirm_seconds = float(seating_cfg.get("initial_confirm_seconds", 3.0))
        self.shift_confirm_seconds = float(seating_cfg.get("shift_confirm_seconds", 10.0))
        self.out_of_class_seconds = float(event_cfg.get("out_of_class_seconds", 20.0))
        self.exit_zone_seconds = float(event_cfg.get("exit_zone_seconds", 8.0))
        self.late_arrival_minutes = float(event_cfg.get("late_arrival_minutes", 5.0))
        self.early_exit_minutes = float(event_cfg.get("early_exit_minutes", 5.0))

        self._cached_pose: dict[int, tuple[int, dict]] = {}
        self._cached_phone_detections: tuple[int, list[dict]] = (-10**9, [])

    @staticmethod
    def _cadence_frames(video_fps: float, target_fps: float) -> int:
        if target_fps <= 0:
            return 1
        return max(1, int(round(float(video_fps) / max(1e-6, float(target_fps)))))

    def _setup_seat_system(self, video_path: str, fps: float) -> None:
        if not self.seat_calibration_path:
            return
        self._seat_calibration = load_seat_calibration(self.seat_calibration_path)
        self._seat_map = build_seat_map(self._seat_calibration)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video for calibration: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(self._seat_calibration.reference_frame_index)))
        ok, reference_frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Could not read calibration reference frame from video.")
        self._reference_frame = reference_frame
        self._projection_manager = CameraMotionCompensator(reference_frame, self._seat_calibration, self._seat_map)
        self._seat_event_engine = SeatEventEngine(
            self._seat_map,
            fps=fps,
            initial_confirm_seconds=self.initial_confirm_seconds,
            shift_confirm_seconds=self.shift_confirm_seconds,
            out_of_class_seconds=self.out_of_class_seconds,
            exit_zone_seconds=self.exit_zone_seconds,
            late_arrival_minutes=self.late_arrival_minutes,
            early_exit_minutes=self.early_exit_minutes,
        )
        self._hand_raise_tracker = HandRaiseEventTracker(fps=fps)

    def process_video(self, video_path: str, output_path: str | None = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ok, first_frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Cannot read first frame.")
        height, width = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if self.seat_calibration_path:
            self._setup_seat_system(video_path, fps)

        os.makedirs(self.output_dir, exist_ok=True)
        writer = None
        if self.save_video:
            if output_path is None:
                output_path = os.path.join(self.output_dir, "output.avi")
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"MJPG"),
                fps,
                (width, height),
            )

        processing_start = datetime.now()
        pose_every = self._cadence_frames(fps, self.pose_fps)
        phone_every = self._cadence_frames(fps, self.phone_object_fps)
        frame_idx = 0

        try:
            with tqdm(total=total_frames, desc="Processing", unit="frame", ncols=100) as progress:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_idx += 1
                    if self.frame_skip > 1 and frame_idx % self.frame_skip != 0:
                        if writer is not None:
                            writer.write(frame)
                        progress.update(1)
                        continue

                    annotated = self._process_frame(frame, frame_idx, fps, pose_every, phone_every)
                    if writer is not None:
                        writer.write(annotated)
                    progress.update(1)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            self.student_backbone.close()

        if self._seat_event_engine is not None:
            self._seat_event_engine.finalize(frame_idx)
        if self._hand_raise_tracker is not None:
            self._hand_raise_tracker.finalize(frame_idx)

        processing_time = (datetime.now() - processing_start).total_seconds()
        if self.save_csv:
            self._save_outputs()
        self._print_session_summary(video_path, frame_idx, processing_time, fps)

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        pose_every: int,
        phone_every: int,
    ) -> np.ndarray:
        body_detections = self.person_detector.detect(frame)
        observations = self.student_backbone.step(frame, frame_idx, fps, body_detections=body_detections)

        seat_assignments: dict[str, str | None] = {}
        seat_states: dict[str, str] = {}
        projection = None
        if self._projection_manager is not None and self._seat_event_engine is not None:
            projection = self._projection_manager.project(frame, frame_idx)
            seat_students = []
            for obs in observations:
                ref_box = self._reference_bbox(obs)
                seat_students.append(
                    {
                        "global_id": obs.global_id,
                        "track_id": obs.track_id,
                        "center": _box_center(ref_box).tolist(),
                    }
                )
            seat_assignments = self._seat_event_engine.update(frame_idx, projection, seat_students)
            seat_states = {
                obs.global_id: self._seat_event_engine.get_current_state(obs.global_id)
                for obs in observations
            }

        if frame_idx % phone_every == 0:
            self._cached_phone_detections = (frame_idx, self.phone_detector.detect(frame))
        phone_detections = self._cached_phone_detections[1]

        pose_targets = [
            {"track_id": obs.track_id, "bbox": obs.body_bbox}
            for obs in observations
            if obs.body_bbox is not None and obs.size_mode != "limited"
        ]
        if pose_targets and frame_idx % pose_every == 0:
            pose_results = self.pose_analyzer.analyze_batch(frame, pose_targets)
            for track_id, pose_result in pose_results.items():
                self._cached_pose[int(track_id)] = (frame_idx, pose_result)

        for obs in observations:
            seat_id = seat_assignments.get(obs.global_id)
            seat_state = seat_states.get(obs.global_id, "unassigned")
            self.attendance_manager.update(obs.global_id, obs.track_id, frame_idx, fps)
            pose_result = self._get_pose_result(obs.track_id, frame_idx, pose_every)
            phone_match = self._match_phone_evidence(obs, phone_detections)

            if self._hand_raise_tracker is not None:
                self._hand_raise_tracker.update(
                    obs.global_id,
                    obs.track_id,
                    frame_idx,
                    bool(pose_result.get("hand_raised", False)),
                    float(pose_result.get("pose_confidence", 0.0)),
                    seat_id or "",
                )

            signals = {
                "hand_raised": pose_result.get("hand_raised", False),
                "head_forward": pose_result.get("head_forward", False),
                "using_phone_pose": pose_result.get("using_phone_pose", False),
                "using_phone_object": phone_match["matched"],
                "phone_object_confidence": phone_match["confidence"],
                "pose_confidence": pose_result.get("pose_confidence", 0.0),
                "detection_confidence": max(obs.detection_confidence, obs.face_confidence),
                "size_mode": obs.size_mode,
            }
            attention = self.attention_engine.update(obs.global_id, signals)
            frame = self._draw_annotations(frame, obs, attention, pose_result, phone_match, seat_id, seat_state)

        if projection is not None:
            frame = self._draw_seat_overlay(frame, projection, seat_assignments)
        return frame

    def _draw_seat_overlay(self, frame: np.ndarray, projection, seat_assignments: dict[str, str | None]) -> np.ndarray:
        occupied = {seat_id for seat_id in seat_assignments.values() if seat_id}
        for seat in self._seat_map:
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

    def _get_pose_result(self, track_id: int, frame_idx: int, pose_every: int) -> dict:
        cached = self._cached_pose.get(int(track_id))
        if cached is None:
            return self.pose_analyzer._default_result()
        if frame_idx - int(cached[0]) > max(2, pose_every * 2):
            return self.pose_analyzer._default_result()
        return cached[1]

    @staticmethod
    def _reference_bbox(obs: StudentObservation) -> list[float]:
        return list(obs.body_bbox if obs.body_bbox is not None else obs.face_bbox)

    def _match_phone_evidence(self, obs: StudentObservation, phone_detections: list[dict]) -> dict:
        if not phone_detections:
            return {"matched": False, "confidence": 0.0, "bbox": None}

        ref_box = self._reference_bbox(obs)
        rx1, ry1, rx2, ry2 = [float(v) for v in ref_box]
        ref_w = max(1.0, rx2 - rx1)
        ref_h = max(1.0, ry2 - ry1)
        ref_cx = 0.5 * (rx1 + rx2)
        ref_cy = 0.5 * (ry1 + ry2)

        face_x1, face_y1, face_x2, face_y2 = [float(v) for v in obs.face_bbox]
        face_size = max(1.0, max(face_x2 - face_x1, face_y2 - face_y1))
        face_cx = 0.5 * (face_x1 + face_x2)
        face_cy = 0.5 * (face_y1 + face_y2)

        best = {"matched": False, "confidence": 0.0, "bbox": None}
        for det in phone_detections:
            bbox = det["bbox"]
            iou = bbox_iou(ref_box, bbox)
            ox1, oy1, ox2, oy2 = [float(v) for v in bbox]
            ocx = 0.5 * (ox1 + ox2)
            ocy = 0.5 * (oy1 + oy2)
            body_dist = np.hypot(ref_cx - ocx, ref_cy - ocy)
            face_dist = np.hypot(face_cx - ocx, face_cy - ocy)
            close_to_body = max(0.0, 1.0 - body_dist / max(1.0, 0.9 * max(ref_w, ref_h)))
            close_to_face = max(0.0, 1.0 - face_dist / max(1.0, 0.9 * face_size))
            upper_body_bonus = 1.0 if ocy <= (ry1 + 0.55 * ref_h) else 0.0
            score = (
                0.28 * float(det.get("confidence", 0.0))
                + 0.30 * close_to_face
                + 0.22 * close_to_body
                + 0.10 * upper_body_bonus
                + 0.10 * min(1.0, iou * 3.0)
            )
            if score > best["confidence"]:
                best = {
                    "matched": score >= 0.38,
                    "confidence": float(score),
                    "bbox": bbox,
                }
        return best

    def _draw_annotations(
        self,
        frame: np.ndarray,
        obs: StudentObservation,
        attention: dict,
        pose_result: dict,
        phone_match: dict,
        seat_id: str | None,
        seat_state: str,
    ) -> np.ndarray:
        box = self._reference_bbox(obs)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        state = attention["attention_state"]
        color_map = {
            "attentive": (0, 200, 0),
            "distracted": (0, 0, 220),
            "unknown": (140, 140, 140),
        }
        color = color_map.get(state, (255, 255, 255))
        if pose_result.get("hand_raised", False):
            color = (255, 255, 0)
        if phone_match.get("matched", False):
            color = (0, 64, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        seat_text = seat_id or "--"
        label = f"{obs.global_id} {seat_text} | {state.upper()} {attention['attention_confidence']:.2f}"
        sublabel = f"{attention['attention_mode']} | {seat_state}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 18)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, sublabel, (x1, min(frame.shape[0] - 6, y2 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if obs.body_bbox is None:
            fx1, fy1, fx2, fy2 = [int(round(v)) for v in obs.face_bbox]
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 255, 255), 1)
        if phone_match.get("matched", False) and phone_match.get("bbox") is not None:
            px1, py1, px2, py2 = [int(round(v)) for v in phone_match["bbox"]]
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 1)
        return frame

    def _save_outputs(self):
        attendance_csv_path = os.path.join(self.output_dir, "attendance_report.csv")
        self._save_attendance_report(attendance_csv_path)
        if self._seat_calibration is not None and self._reference_frame is not None:
            save_seat_map_json(
                self._seat_map,
                self._seat_calibration,
                os.path.join(self.output_dir, "seat_map.json"),
            )
            save_seat_map_png(
                self._reference_frame,
                self._seat_calibration,
                self._seat_map,
                os.path.join(self.output_dir, "seat_map.png"),
            )
            self._save_seating_timeline(os.path.join(self.output_dir, "student_seating_timeline.csv"))
            self._save_attendance_events(os.path.join(self.output_dir, "attendance_events.csv"))

    def _save_attendance_report(self, csv_path: str):
        attendance_data = self.attendance_manager.get_attendance_report()
        attention_data = self.attention_engine.get_all_metrics()
        seating_summary = self._seat_event_engine.get_student_summary() if self._seat_event_engine else {}
        hand_raise_summary = self._hand_raise_tracker.get_student_summary() if self._hand_raise_tracker else {}
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Session_ID",
                    "Camera_ID",
                    "Global_Student_ID",
                    "Local_Track_ID",
                    "Total_Frames",
                    "Presence_Time_Seconds",
                    "Present",
                    "Seat_ID",
                    "Seat_Rank",
                    "Row_Rank",
                    "Weighted_Avg_Seat_Rank",
                    "Out_of_Class_Seconds",
                    "Late_Arrival",
                    "Early_Exit",
                    "Hand_Raise_Count",
                    "Hand_Raise_Seconds",
                    "Attention_State",
                    "Attention_Confidence",
                    "Attention_Mode",
                    "Attention_Reason",
                    "Attentive_Frames",
                    "Distracted_Frames",
                    "Unknown_Frames",
                    "HandRaise_Frames",
                    "UsingPhone_Frames",
                    "Confidence_Weighted_Attention_Score",
                    "Attention_Percentage",
                ]
            )
            for record in attendance_data:
                student_id = record["global_id"]
                metrics = attention_data.get(
                    student_id,
                    {
                        "attention_state": "unknown",
                        "attention_confidence": 0.0,
                        "attention_mode": "limited",
                        "attention_reason": "no-observations",
                        "attentive_frames": 0,
                        "distracted_frames": 0,
                        "unknown_frames": 0,
                        "handraise_frames": 0,
                        "using_phone_frames": 0,
                        "confidence_weighted_score": 0.0,
                        "attention_percentage": 0.0,
                    },
                )
                seat_info = seating_summary.get(student_id, {})
                hand_info = hand_raise_summary.get(student_id, {"hand_raise_count": 0, "hand_raise_seconds": 0.0})
                writer.writerow(
                    [
                        self.session_id,
                        self.camera_id,
                        student_id,
                        record["local_track_id"],
                        record["total_frames"],
                        record["presence_time_seconds"],
                        "Yes" if record["is_present"] else "No",
                        seat_info.get("seat_id") or "",
                        seat_info.get("seat_rank") or "",
                        seat_info.get("row_rank") or "",
                        seat_info.get("weighted_avg_seat_rank") or "",
                        seat_info.get("out_of_class_seconds") or 0.0,
                        "Yes" if seat_info.get("late_arrival") else "No",
                        "Yes" if seat_info.get("early_exit") else "No",
                        hand_info.get("hand_raise_count", 0),
                        hand_info.get("hand_raise_seconds", 0.0),
                        metrics["attention_state"],
                        metrics["attention_confidence"],
                        metrics["attention_mode"],
                        metrics["attention_reason"],
                        metrics["attentive_frames"],
                        metrics["distracted_frames"],
                        metrics["unknown_frames"],
                        metrics["handraise_frames"],
                        metrics["using_phone_frames"],
                        metrics["confidence_weighted_score"],
                        metrics["attention_percentage"],
                    ]
                )
        logger.info("Attendance report saved: %s", csv_path)

    def _save_seating_timeline(self, csv_path: str):
        if self._seat_event_engine is None:
            return
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
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
            writer.writeheader()
            writer.writerows(self._seat_event_engine.get_timeline_rows())

    def _save_attendance_events(self, csv_path: str):
        if self._seat_event_engine is None:
            return
        rows = self._seat_event_engine.get_event_rows()
        if self._hand_raise_tracker is not None:
            rows.extend(self._hand_raise_tracker.get_event_rows())
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else [
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
                "peak_confidence",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _print_session_summary(
        self,
        video_path: str,
        total_frames: int,
        processing_time: float,
        fps: float,
    ):
        attendance_summary = self.attendance_manager.get_summary()
        attention_metrics = self.attention_engine.get_all_metrics()
        if attention_metrics:
            avg_attention = float(np.mean([metrics["attention_percentage"] for metrics in attention_metrics.values()]))
            avg_unknown = float(
                np.mean(
                    [
                        (metrics["unknown_frames"] / max(1, metrics["total_frames"])) * 100.0
                        for metrics in attention_metrics.values()
                    ]
                )
            )
        else:
            avg_attention = 0.0
            avg_unknown = 0.0

        processing_fps = total_frames / processing_time if processing_time > 0 else 0.0
        print(f"\n{'=' * 60}")
        print("  SESSION SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Session ID      : {self.session_id}")
        print(f"  Camera ID       : {self.camera_id}")
        print(f"  Video           : {video_path}")
        print(f"  Total Frames    : {total_frames}")
        print(f"  Video Duration  : {total_frames / max(1e-6, fps):.1f}s")
        print(f"  Processing Time : {processing_time:.1f}s")
        print(f"  Processing FPS  : {processing_fps:.1f}")
        print(f"  Students Tracked: {attendance_summary['total_students']}")
        print(f"  Present         : {attendance_summary['present_count']}")
        print(f"  Avg Attention   : {avg_attention:.1f}%")
        print(f"  Avg Unknown     : {avg_unknown:.1f}%")
        if self._seat_event_engine is not None:
            print(f"  Seat Map        : enabled ({len(self._seat_map)} seats)")
        print(f"{'=' * 60}\n")

