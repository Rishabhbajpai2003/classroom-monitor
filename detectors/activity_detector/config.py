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
PEN_LIKE = {"pen", "pencil", "stylus", "marker", "highlighter"}
NOTE_SURFACE_LIKE = {
    "notebook",
    "paper",
    "sheet of paper",
    "document",
    "handout",
    "book",
    "worksheet",
    "writing",
    "notes",
}
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recall-first classroom activity detector with size-aware confidence outputs."
    )
    parser.add_argument("--weights", default="weights/yolov8x-worldv2.pt", help="Open-vocabulary object detector weights.")
    parser.add_argument("--pose_weights", default="weights/yolo11x-pose.pt", help="Pose model weights.")
    parser.add_argument("--source", required=True, help="Input video path or YouTube URL.")
    parser.add_argument("--out", required=True, help="Annotated output video path.")
    parser.add_argument("--device", default=None, help="Inference device (e.g. 0, cuda, cpu).")

    parser.add_argument("--electronics", default=",".join(DEFAULT_ELECTRONICS), help="Comma-separated electronics classes.")
    parser.add_argument("--notetaking", default=",".join(DEFAULT_NOTE_TAKING), help="Comma-separated note-taking classes.")

    parser.add_argument("--person_conf", type=float, default=0.22, help="Person detector confidence threshold.")
    parser.add_argument("--object_conf", type=float, default=0.14, help="Object detector confidence threshold.")
    parser.add_argument("--pose_conf", type=float, default=0.18, help="Pose model confidence threshold.")
    parser.add_argument("--person_imgsz", type=int, default=1536, help="Person detector image size.")
    parser.add_argument("--object_imgsz", type=int, default=1280, help="Object detector image size.")
    parser.add_argument("--pose_imgsz", type=int, default=1152, help="Pose detector image size.")
    parser.add_argument("--tile_grid", type=int, default=2, help="Tile grid for person/object detection.")
    parser.add_argument("--tile_overlap", type=float, default=0.20, help="Tile overlap ratio.")

    parser.add_argument("--track_fps", type=float, default=9.0, help="Face/body tracking rate.")
    parser.add_argument("--object_fps", type=float, default=3.0, help="Object detection rate.")
    parser.add_argument("--pose_fps", type=float, default=2.5, help="Pose refresh rate.")

    parser.add_argument("--face_det_size", type=int, default=1600, help="InsightFace detection size.")
    parser.add_argument("--face_tile_grid", type=int, default=3, help="InsightFace tile grid.")
    parser.add_argument("--face_tile_overlap", type=float, default=0.22, help="InsightFace tile overlap.")
    parser.add_argument("--min_face", type=int, default=12, help="Minimum face size in pixels.")
    parser.add_argument("--primary_detector", default="scrfd", help="Primary detector label used in metadata.")
    parser.add_argument("--backup_detector", default="retinaface", help="Backup detector label used in metadata.")
    parser.add_argument("--disable_backup_detector", action="store_true", help="Disable the backup detector pass.")
    parser.add_argument("--adaface_weights", default="", help="Optional AdaFace ONNX weights.")
    parser.add_argument("--identity_db", default="detectors/face_detector/identity_db.json", help="Persistent face identity DB.")
    parser.add_argument("--identity_db_save_every", type=int, default=150, help="Save identity DB every N frames.")
    parser.add_argument("--seat_calibration", default=None, help="Optional seat calibration JSON path.")

    parser.add_argument("--full_min_height", type=int, default=160, help="Body height threshold for full mode.")
    parser.add_argument("--reduced_min_height", type=int, default=96, help="Body height threshold for reduced mode.")
    parser.add_argument("--full_conf_threshold", type=float, default=0.52, help="Proof threshold for full mode.")
    parser.add_argument("--reduced_conf_threshold", type=float, default=0.60, help="Proof threshold for reduced mode.")
    parser.add_argument("--limited_conf_threshold", type=float, default=0.72, help="Proof threshold for limited mode.")

    parser.add_argument("--fps_out", type=float, default=0.0, help="Annotated output FPS. <= 0 keeps source FPS.")
    parser.add_argument("--activity_out", default="person_activity_summary.csv", help="CSV output path.")
    parser.add_argument("--proof_dir", default="proof_keyframes", help="Directory for proof keyframes.")
    parser.add_argument("--download_dir", default="_downloads", help="Directory for downloaded source videos.")
    parser.add_argument("--keep_downloaded_source", action="store_true", help="Keep downloaded video when using URL input.")

    parser.add_argument("--hands", action="store_true", help="Enable ROI-only MediaPipe hand refinement.")
    parser.add_argument("--max_hands", type=int, default=4, help="Maximum hands for ROI refinement.")
    parser.add_argument("--hands_conf", type=float, default=0.5, help="MediaPipe hand detection confidence.")
    parser.add_argument("--hands_track", type=float, default=0.5, help="MediaPipe hand tracking confidence.")

    parser.add_argument("--phone_conf", type=float, default=0.45, help="Minimum confidence for phone activity assignment.")
    parser.add_argument("--laptop_conf", type=float, default=0.48, help="Minimum confidence for laptop activity assignment.")
    parser.add_argument("--tablet_conf", type=float, default=0.46, help="Minimum confidence for tablet activity assignment.")
    parser.add_argument("--note_conf", type=float, default=0.46, help="Minimum confidence for note-taking assignment.")
    parser.add_argument("--idle_conf", type=float, default=0.35, help="Confidence needed for an explicit idle label.")
    parser.add_argument("--max_frames", type=int, default=-1, help="Optional frame cap for debugging.")

    return parser.parse_args()
