"""
Main processing pipeline for classroom attention and attendance monitoring.

Integrates:
    - YOLO26 person detection
    - DeepSORT tracking with ReID
    - Face detection and embedding extraction
    - Pose analysis (hand raise, head forward, phone usage)
    - Persistent identity matching
    - Attendance tracking
    - Confidence-weighted attention scoring
    - Annotated video output
    - Structured CSV report generation
"""

import csv
import logging
import os
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

from app.models.detectors import PersonDetector
from app.models.tracker import PersonTracker
from app.models.pose_analyzer import PoseAnalyzer
from app.models.face_detector import FaceDetector
from app.models.identity_manager import IdentityManager
from app.models.attendance_manager import AttendanceManager
from app.fusion.attention_engine import AttentionEngine

logger = logging.getLogger(__name__)


class ClassroomProcessor:
    """
    Production-grade classroom attendance and attention processing pipeline.

    Processes a video through detection → tracking → identity → pose analysis
    → attention scoring → attendance, producing annotated video and CSV reports.

    Attributes:
        config: Pipeline configuration dict.
        camera_id: Camera identifier string.
        session_id: Unique session identifier.
        person_detector: YOLO26 person detector.
        tracker: DeepSORT tracker with ReID.
        face_detector: Face detector with embedding extraction.
        pose_analyzer: MediaPipe pose analyzer.
        identity_manager: Persistent student identity matcher.
        attendance_manager: Attendance tracker.
        attention_engine: Confidence-weighted attention scorer.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize the ClassroomProcessor.

        Args:
            config: Full pipeline configuration dict (from config.yaml).
        """
        self.config = config or {}
        sys_cfg = self.config.get("system", {})

        self.camera_id = sys_cfg.get("camera_id", "cam_01")
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.camera_id}"
        self.output_dir = sys_cfg.get("output_dir", "outputs")
        self.save_video = sys_cfg.get("save_video", True)
        self.save_csv = sys_cfg.get("save_csv", True)
        self.headless = sys_cfg.get("headless", False)
        self.frame_skip = sys_cfg.get("frame_skip", 2)
        self.pose_skip = sys_cfg.get("pose_skip", 5)

        logger.info(f"Initializing ClassroomProcessor — Session: {self.session_id}")

        # Initialize all pipeline components
        self.person_detector = PersonDetector(self.config)
        self.tracker = PersonTracker(self.config)
        self.face_detector = FaceDetector(self.config)
        self.pose_analyzer = PoseAnalyzer(self.config)
        self.identity_manager = IdentityManager(self.config)
        self.attendance_manager = AttendanceManager(self.config)
        self.attention_engine = AttentionEngine(self.config)

        # Pose cache per track (to avoid running pose every frame)
        self._pose_cache: dict[int, dict] = {}

        # Track embedding extraction schedule (extract once per track)
        self._embedding_extracted: set[int] = set()

        logger.info("All pipeline components initialized successfully.")

    def process_video(self, video_path: str, output_path: str | None = None):
        """
        Process a video through the full pipeline.

        Args:
            video_path: Path to input video file.
            output_path: Path for annotated output video. If None, auto-generated.
        """
        # --- Video setup ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 30.0
            logger.warning(f"Could not read FPS, defaulting to {fps}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first_frame = cap.read()
        if not ret:
            logger.error("Cannot read first frame.")
            cap.release()
            raise RuntimeError("Cannot read first frame.")

        h, w = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        logger.info(
            f"Video: {video_path} | {w}x{h} @ {fps:.1f} FPS | "
            f"{total_frames} frames | Camera: {self.camera_id}"
        )

        # --- Video writer setup ---
        os.makedirs(self.output_dir, exist_ok=True)
        out_writer = None
        if self.save_video:
            if output_path is None:
                output_path = os.path.join(self.output_dir, "output.avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            logger.info(f"Output video: {output_path}")

        # --- Processing loop ---
        frame_count = 0
        processing_start = datetime.now()

        print(f"\n{'='*60}")
        print(f"  Classroom Monitoring Pipeline — Session: {self.session_id}")
        print(f"  Video: {video_path}")
        print(f"  Camera: {self.camera_id}")
        print(f"  Total frames: {total_frames}")
        print(f"{'='*60}\n")

        try:
            with tqdm(total=total_frames, desc="Processing", unit="frame",
                       ncols=100) as pbar:

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # Frame skip optimization
                    if frame_count % self.frame_skip != 0:
                        if out_writer:
                            out_writer.write(frame)
                        pbar.update(1)
                        continue

                    # Process this frame
                    annotated_frame = self._process_frame(
                        frame, frame_count, fps
                    )

                    if out_writer:
                        out_writer.write(annotated_frame)

                    pbar.update(1)

        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user.")
        except Exception as e:
            logger.error(f"Processing error at frame {frame_count}: {e}")
            raise
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            logger.info("Video resources released.")

        processing_time = (datetime.now() - processing_start).total_seconds()

        # --- Generate outputs ---
        if self.save_csv:
            csv_path = os.path.join(self.output_dir, "attendance_report.csv")
            self._save_attendance_report(csv_path)

        # --- Print session summary ---
        self._print_session_summary(
            video_path, frame_count, processing_time, fps
        )

    def _process_frame(self, frame: np.ndarray, frame_idx: int,
                       fps: float) -> np.ndarray:
        """
        Process a single frame through the full pipeline.

        Args:
            frame: BGR frame (numpy array).
            frame_idx: Current frame index.
            fps: Video FPS.

        Returns:
            Annotated frame with bounding boxes and status labels.
        """
        # 1. Person detection (YOLO26)
        detections = self.person_detector.detect(frame)

        # 2. Format detections for DeepSORT: ([x, y, w, h], confidence, class)
        tracker_dets = []
        det_confidence_map = {}

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            w_box = x2 - x1
            h_box = y2 - y1
            conf = det["confidence"]
            tracker_dets.append(
                ([float(x1), float(y1), float(w_box), float(h_box)],
                 conf, "person")
            )
            det_confidence_map[i] = conf

        # 3. Update tracker
        tracks = self.tracker.update(tracker_dets, frame)

        # 4. Collect all confirmed tracks first (for batch pose)
        confirmed_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            person_bbox = [l, t, r, b]

            det_conf = self._get_track_detection_confidence(
                person_bbox, detections
            )

            # Identity matching (once per track)
            if track_id not in self._embedding_extracted:
                embedding = self.face_detector.extract_embedding(
                    frame, person_bbox
                )
                timestamp = datetime.now().isoformat()
                global_id = self.identity_manager.get_global_id_for_track(
                    track_id, embedding, timestamp
                )
                self._embedding_extracted.add(track_id)
            else:
                global_id = self.identity_manager.get_global_id_for_track(
                    track_id
                )

            # Attendance update
            self.attendance_manager.update(
                global_id, track_id, frame_idx, fps
            )

            confirmed_tracks.append({
                "track_id": track_id,
                "bbox": person_bbox,
                "det_conf": det_conf,
                "global_id": global_id,
            })

        # 5. Batch pose analysis — ONE YOLO inference for ALL persons
        if frame_idx % self.pose_skip == 0 and confirmed_tracks:
            pose_results = self.pose_analyzer.analyze_batch(
                frame,
                [{"track_id": t["track_id"], "bbox": t["bbox"]}
                 for t in confirmed_tracks],
            )
            # Update cache
            for tid, result in pose_results.items():
                self._pose_cache[tid] = result

        # 6. Attention scoring + annotation for each person
        for info in confirmed_tracks:
            track_id = info["track_id"]
            person_bbox = info["bbox"]
            global_id = info["global_id"]
            det_conf = info["det_conf"]

            pose_result = self._pose_cache.get(track_id, {
                "hand_raised": False,
                "head_forward": False,
                "using_phone": False,
                "pose_confidence": 0.0,
            })

            signals = {
                "hand_raised": pose_result["hand_raised"],
                "head_forward": pose_result["head_forward"],
                "using_phone": pose_result["using_phone"],
                "pose_confidence": pose_result["pose_confidence"],
                "detection_confidence": det_conf,
            }
            attention_status = self.attention_engine.update(global_id, signals)

            frame = self._draw_annotations(
                frame, person_bbox, track_id, global_id,
                attention_status, pose_result, det_conf
            )

        return frame

    def _get_track_detection_confidence(
        self, track_bbox: list, detections: list[dict]
    ) -> float:
        """
        Find the detection confidence that best matches a track's bbox
        using IoU overlap.
        """
        if not detections:
            return 0.5

        best_iou = 0.0
        best_conf = 0.5

        tl, tt, tr, tb = track_bbox

        for det in detections:
            dl, dt, dr, db = det["bbox"]

            # Compute IoU
            inter_l = max(tl, dl)
            inter_t = max(tt, dt)
            inter_r = min(tr, dr)
            inter_b = min(tb, db)

            if inter_r <= inter_l or inter_b <= inter_t:
                continue

            inter_area = (inter_r - inter_l) * (inter_b - inter_t)
            track_area = max(0, (tr - tl) * (tb - tt))
            det_area = max(0, (dr - dl) * (db - dt))
            union = track_area + det_area - inter_area

            if union > 0:
                iou = inter_area / union
                if iou > best_iou:
                    best_iou = iou
                    best_conf = det["confidence"]

        return best_conf

    def _draw_annotations(
        self, frame: np.ndarray, bbox: list,
        track_id: int, global_id: str, status: str,
        pose_result: dict, det_conf: float
    ) -> np.ndarray:
        """Draw bounding box, ID labels, and status on the frame."""
        l, t, r, b = map(int, bbox)

        # Color based on attention status
        color_map = {
            "attentive": (0, 255, 0),       # Green
            "distracted": (0, 0, 255),       # Red
            "neutral": (0, 165, 255),        # Orange
        }
        color = color_map.get(status, (255, 255, 255))

        # Special color for hand raised
        if pose_result.get("hand_raised"):
            color = (255, 255, 0)  # Cyan
        if pose_result.get("using_phone"):
            color = (0, 0, 200)    # Dark red

        # Draw bounding box
        cv2.rectangle(frame, (l, t), (r, b), color, 2)

        # Status label
        status_text = status.upper()
        if pose_result.get("hand_raised"):
            status_text = "HAND RAISED"
        elif pose_result.get("using_phone"):
            status_text = "USING PHONE"

        # ID and status label
        label = f"{global_id} | {status_text}"
        conf_label = f"conf: {det_conf:.2f}"

        # Background for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(frame, (l, t - th - 10), (l + tw + 4, t), color, -1)
        cv2.putText(frame, label, (l + 2, t - 5),
                    font, font_scale, (0, 0, 0), thickness)

        # Confidence below box
        cv2.putText(frame, conf_label, (l, b + 15),
                    font, 0.4, color, 1)

        return frame

    def _save_attendance_report(self, csv_path: str):
        """
        Save combined attendance and attention report to CSV.

        Columns:
            Session_ID, Camera_ID, Global_Student_ID, Local_Track_ID,
            Total_Frames, Presence_Time_Seconds, Present,
            Attentive_Frames, Distracted_Frames, HandRaise_Frames,
            UsingPhone_Frames, Confidence_Weighted_Attention_Score,
            Attention_Percentage
        """
        attendance_data = self.attendance_manager.get_attendance_report()
        attention_data = self.attention_engine.get_all_metrics()

        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Session_ID",
                "Camera_ID",
                "Global_Student_ID",
                "Local_Track_ID",
                "Total_Frames",
                "Presence_Time_Seconds",
                "Present",
                "Attentive_Frames",
                "Distracted_Frames",
                "HandRaise_Frames",
                "UsingPhone_Frames",
                "Confidence_Weighted_Attention_Score",
                "Attention_Percentage",
            ])

            for record in attendance_data:
                gid = record["global_id"]
                metrics = attention_data.get(gid, {
                    "attentive_frames": 0,
                    "distracted_frames": 0,
                    "handraise_frames": 0,
                    "using_phone_frames": 0,
                    "confidence_weighted_score": 0.0,
                    "attention_percentage": 0.0,
                })

                writer.writerow([
                    self.session_id,
                    self.camera_id,
                    gid,
                    record["local_track_id"],
                    record["total_frames"],
                    record["presence_time_seconds"],
                    "Yes" if record["is_present"] else "No",
                    metrics.get("attentive_frames", 0),
                    metrics.get("distracted_frames", 0),
                    metrics.get("handraise_frames", 0),
                    metrics.get("using_phone_frames", 0),
                    metrics.get("confidence_weighted_score", 0.0),
                    metrics.get("attention_percentage", 0.0),
                ])

        logger.info(f"Attendance report saved: {csv_path}")
        print(f"\n📄 Attendance report saved: {csv_path}")

    def _print_session_summary(
        self, video_path: str, total_frames: int,
        processing_time: float, fps: float
    ):
        """Print a comprehensive session summary to console."""
        attendance_summary = self.attendance_manager.get_summary()
        attention_metrics = self.attention_engine.get_all_metrics()

        # Compute overall attention average
        if attention_metrics:
            avg_attention = np.mean([
                m["attention_percentage"]
                for m in attention_metrics.values()
            ])
        else:
            avg_attention = 0.0

        processing_fps = total_frames / processing_time if processing_time > 0 else 0

        print(f"\n{'='*60}")
        print(f"  📊 SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"  Session ID     : {self.session_id}")
        print(f"  Camera ID      : {self.camera_id}")
        print(f"  Video          : {video_path}")
        print(f"  Total Frames   : {total_frames}")
        print(f"  Video Duration : {total_frames / fps:.1f}s")
        print(f"  Processing Time: {processing_time:.1f}s")
        print(f"  Processing FPS : {processing_fps:.1f}")
        print(f"{'─'*60}")
        print(f"  👥 ATTENDANCE")
        print(f"     Total Students Detected : {attendance_summary['total_students']}")
        print(f"     Present                 : {attendance_summary['present_count']}")
        print(f"     Absent / Short Tracks   : {attendance_summary['absent_count']}")
        print(f"{'─'*60}")
        print(f"  🧠 ATTENTION")
        print(f"     Average Attention       : {avg_attention:.1f}%")
        print(f"     Students Tracked        : {len(attention_metrics)}")
        print(f"{'─'*60}")
        print(f"  📁 OUTPUTS")
        print(f"     Video  : {self.output_dir}/output.avi")
        print(f"     CSV    : {self.output_dir}/attendance_report.csv")
        print(f"     Registry: {self.identity_manager.registry_path}")
        print(f"{'='*60}\n")

        logger.info(
            f"Session complete: {attendance_summary['total_students']} students, "
            f"{attendance_summary['present_count']} present, "
            f"avg attention {avg_attention:.1f}%"
        )