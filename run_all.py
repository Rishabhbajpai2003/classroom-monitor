from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2

from app.utils.reporting import build_final_student_summary, build_run_manifest


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_TOPIC_CONFIG = PROJECT_ROOT / "configs" / "topic_profiles.yaml"
DEFAULT_IDENTITY_DB = PROJECT_ROOT / "detectors" / "face_detector" / "identity_db.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full classroom monitoring stack into a single run folder.")
    parser.add_argument("--source", required=True, help="Input classroom video path.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Main classroom config path.")
    parser.add_argument("--device", default=None, help="Compute device override (auto/cuda/cpu).")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory for this run.")
    parser.add_argument("--seat-calibration", default=None, help="Optional seat calibration JSON.")
    parser.add_argument("--topic-config", default=str(DEFAULT_TOPIC_CONFIG), help="Speech topic profile config.")
    parser.add_argument("--course-profile", default="default", help="Topic profile name inside the topic config.")
    parser.add_argument("--skip-face", action="store_true", help="Skip the standalone face identity run.")
    parser.add_argument("--skip-speech", action="store_true", help="Skip VSD and VLP runs.")
    parser.add_argument("--max-frames", type=int, default=-1, help="Optional frame cap for debugging.")
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def run_command(cmd: list[str], description: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"Starting: {description}")
    print("Command :", " ".join(cmd))
    print(f"{'=' * 72}")
    subprocess.run(cmd, check=True)
    print(f"Completed: {description}")


def detect_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 0 else 0.0


def main() -> None:
    args = parse_args()
    python_exe = sys.executable
    source_path = Path(args.source).resolve()
    ensure_exists(source_path, "Source video")

    config_path = Path(args.config).resolve()
    ensure_exists(config_path, "Config")

    topic_config_path = Path(args.topic_config).resolve()
    ensure_exists(topic_config_path, "Topic config")

    seat_calibration_path = Path(args.seat_calibration).resolve() if args.seat_calibration else None
    if seat_calibration_path is not None:
        ensure_exists(seat_calibration_path, "Seat calibration")

    base_output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (PROJECT_ROOT / "outputs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    identity_db_run_path = base_output_dir / "identity_db.json"
    ensure_exists(DEFAULT_IDENTITY_DB, "Identity DB")
    shutil.copy2(DEFAULT_IDENTITY_DB, identity_db_run_path)

    config_snapshot_path = base_output_dir / Path(config_path).name
    if config_snapshot_path.resolve() != config_path:
        shutil.copy2(config_path, config_snapshot_path)

    if seat_calibration_path is not None:
        calibration_snapshot_path = base_output_dir / Path(seat_calibration_path).name
        if calibration_snapshot_path.resolve() != seat_calibration_path:
            shutil.copy2(seat_calibration_path, calibration_snapshot_path)
        seat_calibration_arg = str(calibration_snapshot_path)
    else:
        seat_calibration_arg = None

    if args.skip_face:
        print("Skipping standalone face identity run.")
    else:
        cmd_face = [
            python_exe,
            "detectors/face_detector/run.py",
            "--input",
            str(source_path),
            "--output",
            str(base_output_dir / "face_identity.mp4"),
            "--csv",
            str(base_output_dir / "face_identity.csv"),
            "--identity-db",
            str(identity_db_run_path),
            "--student-details-root",
            str(PROJECT_ROOT / "student-details"),
            "--unknown-output-dir",
            str(base_output_dir / "unknown_review"),
            "--process-fps",
            "8",
            "--det-size",
            "1280",
            "--tile-grid",
            "2",
            "--tile-overlap",
            "0.20",
            "--min-face",
            "12",
        ]
        run_command(cmd_face, "Face Identity Detector")

    cmd_attention = [
        python_exe,
        "detectors/attention_detector/run.py",
        "--video",
        str(source_path),
        "--config",
        str(config_snapshot_path),
        "--output-dir",
        str(base_output_dir),
        "--headless",
    ]
    if args.device:
        cmd_attention.extend(["--device", args.device])
    if seat_calibration_arg:
        cmd_attention.extend(["--seat-calibration", seat_calibration_arg])
    run_command(cmd_attention, "Attention + Attendance Detector")

    cmd_activity = [
        python_exe,
        "detectors/activity_detector/run.py",
        "--source",
        str(source_path),
        "--out",
        str(base_output_dir / "activity_tracking.mp4"),
        "--activity_out",
        str(base_output_dir / "person_activity_summary.csv"),
        "--proof_dir",
        "proof_keyframes",
        "--identity_db",
        str(identity_db_run_path),
        "--track_fps",
        "8",
    ]
    if args.device:
        cmd_activity.extend(["--device", args.device])
    if seat_calibration_arg:
        cmd_activity.extend(["--seat_calibration", seat_calibration_arg])
    if args.max_frames > 0:
        cmd_activity.extend(["--max_frames", str(args.max_frames)])
    run_command(cmd_activity, "Activity Detector")

    if args.skip_speech:
        print("Skipping VSD and lip-reading runs.")
    else:
        cmd_vsd = [
            python_exe,
            "detectors/vsd_detector/run.py",
            "--input",
            str(source_path),
            "--output",
            str(base_output_dir / "visual_speaking.mp4"),
            "--csv",
            str(base_output_dir / "visual_speaking.csv"),
            "--identity-db",
            str(identity_db_run_path),
            "--process-fps",
            "5",
            "--det-size",
            "1280",
            "--tile-grid",
            "2",
            "--tile-overlap",
            "0.20",
            "--min-face",
            "12",
        ]
        if args.max_frames > 0:
            cmd_vsd.extend(["--max-frames", str(args.max_frames)])
        run_command(cmd_vsd, "Visual Speech Detector")

        cmd_vlp = [
            python_exe,
            "detectors/vlp_detector/run.py",
            "--input",
            str(source_path),
            "--output",
            str(base_output_dir / "speech_topics.mp4"),
            "--csv",
            str(base_output_dir / "speech_topic_segments.csv"),
            "--identity-db",
            str(identity_db_run_path),
            "--topic-config",
            str(topic_config_path),
            "--course-profile",
            args.course_profile,
            "--process-fps",
            "5",
            "--det-size",
            "1280",
            "--tile-grid",
            "2",
            "--tile-overlap",
            "0.20",
            "--min-face",
            "12",
        ]
        if seat_calibration_arg:
            cmd_vlp.extend(["--seat-calibration", seat_calibration_arg])
        if args.max_frames > 0:
            cmd_vlp.extend(["--max-frames", str(args.max_frames)])
        run_command(cmd_vlp, "Lip Reading + Speech Topic Detector")

    fps = detect_video_fps(source_path)
    summary_path = build_final_student_summary(base_output_dir, identity_db_run_path, fps=fps or None)
    manifest_json_path, manifest_csv_path = build_run_manifest(
        base_output_dir,
        source_video=str(source_path),
        identity_db_path=identity_db_run_path,
    )

    print(f"\nRun complete.")
    print(f"Run directory          : {base_output_dir}")
    print(f"Final student summary  : {summary_path}")
    print(f"Run manifest (json)    : {manifest_json_path}")
    print(f"Run manifest (csv)     : {manifest_csv_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nPipeline step failed with exit code {exc.returncode}: {exc.cmd}")
        sys.exit(exc.returncode or 1)
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"\nPipeline failed: {exc}")
        sys.exit(1)
