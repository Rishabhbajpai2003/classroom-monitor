import argparse


DEFAULT_ELECTRONICS = ["cell phone", "tablet", "laptop"]
DEFAULT_NOTE_TAKING = [
    "notebook",
    "paper",
    "sheet of paper",
    "document",
    "handout",
    "pen",
    "pencil",
    "book",
    "worksheet",
    "writing",
]
PEN_LIKE = {"pen", "pencil"}
NOTE_SURFACE_LIKE = {"notebook", "paper", "book", "worksheet", "writing"}
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Classroom detection with per-person ID and activity durations (electronics vs note-taking)."
    )
    p.add_argument(
        "--weights",
        default="weights/yolov8x-worldv2.pt",
        help="Open-vocabulary YOLO weights. Recommended: weights/yolov8x-worldv2.pt",
    )
    p.add_argument(
        "--pose_weights",
        default="weights/yolo11x-pose.pt",
        help="Pose model for person body + skeleton. Recommended for best quality: weights/yolo11x-pose.pt",
    )
    p.add_argument("--source", required=True, help="Input video path or YouTube URL")
    p.add_argument("--out", required=True, help="Output video path")
    p.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold")
    p.add_argument("--pose_conf", type=float, default=0.18, help="Pose model confidence threshold")
    p.add_argument("--imgsz", type=int, default=960, help="YOLO inference size")
    p.add_argument("--pose_imgsz", type=int, default=1152, help="Pose model inference size")
    p.add_argument("--pose_iou", type=float, default=0.85, help="Pose model NMS IoU threshold")
    p.add_argument("--pose_max_det", type=int, default=300, help="Pose model max detections per frame")
    p.add_argument("--device", default=None, help="Device, e.g. 0 or cpu")
    p.add_argument(
        "--electronics",
        default=",".join(DEFAULT_ELECTRONICS),
        help='Comma-separated classes for electronics (e.g. "cell phone,tablet,laptop")',
    )
    p.add_argument(
        "--notetaking",
        default=",".join(DEFAULT_NOTE_TAKING),
        help='Comma-separated classes for non-electronic note-taking (e.g. "notebook,paper,pen,pencil,book")',
    )
    p.add_argument("--fps_out", type=float, default=1.0, help="Output video FPS (kept at 1.0)")
    p.add_argument("--hands", action="store_true", help="Enable hand detection (MediaPipe Hands)")
    p.add_argument("--max_hands", type=int, default=6, help="Max hands to detect per frame")
    p.add_argument("--hands_conf", type=float, default=0.5, help="MediaPipe hands min detection confidence")
    p.add_argument("--hands_track", type=float, default=0.5, help="MediaPipe hands min tracking confidence")
    p.add_argument(
        "--activity_out",
        default="person_activity_summary.csv",
        help="CSV output path with person activity durations",
    )
    p.add_argument(
        "--proof_dir",
        default="proof_keyframes",
        help="Folder to save proof keyframes referenced from the CSV",
    )
    p.add_argument(
        "--download_dir",
        default="_downloads",
        help="Directory for downloaded source videos when --source is a URL",
    )
    p.add_argument(
        "--keep_downloaded_source",
        action="store_true",
        help="Keep downloaded source file when using URL input",
    )
    p.add_argument("--face_match_thresh", type=float, default=0.48, help="Face matching threshold (lower is stricter)")
    p.add_argument(
        "--fallback_face_cos_thresh",
        type=float,
        default=0.22,
        help="Cosine distance threshold for OpenCV fallback face matching",
    )
    p.add_argument("--track_iou_thresh", type=float, default=0.25, help="Fallback ID IoU threshold")
    p.add_argument(
        "--face_model",
        default="hog",
        choices=["hog", "cnn"],
        help="face_recognition face detector model",
    )
    p.add_argument("--head_kpt_conf", type=float, default=0.35, help="Head keypoint confidence threshold")
    p.add_argument("--laptop_obj_conf", type=float, default=0.36, help="Laptop detection confidence threshold")
    p.add_argument("--phone_obj_conf", type=float, default=0.25, help="Phone detection confidence threshold")
    p.add_argument("--tablet_obj_conf", type=float, default=0.33, help="Tablet detection confidence threshold")
    p.add_argument("--pen_obj_conf", type=float, default=0.20, help="Pen/pencil detection confidence threshold")
    p.add_argument("--note_obj_conf", type=float, default=0.14, help="Note-material detection confidence threshold")
    p.add_argument(
        "--contact_dist_px",
        type=float,
        default=18.0,
        help="Max pixel gap to treat two boxes as touching/contact",
    )
    p.add_argument(
        "--laptop_use_conf",
        type=float,
        default=0.55,
        help="Confidence threshold for assigning laptop usage",
    )
    p.add_argument(
        "--phone_use_conf",
        type=float,
        default=0.62,
        help="Confidence threshold for assigning phone usage",
    )
    p.add_argument(
        "--tablet_note_use_conf",
        type=float,
        default=0.62,
        help="Confidence threshold for assigning note-taking on tablet",
    )
    p.add_argument(
        "--tablet_elec_use_conf",
        type=float,
        default=0.58,
        help="Confidence threshold for assigning electronics usage on tablet",
    )
    p.add_argument(
        "--paper_note_use_conf",
        type=float,
        default=0.56,
        help="Confidence threshold for assigning paper note-taking",
    )
    p.add_argument(
        "--phone_head_near_ratio",
        type=float,
        default=0.55,
        help="Normalized distance threshold for head-to-phone proximity",
    )
    return p.parse_args()
