import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 72}\nSMOKE: {label}\n{'=' * 72}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke runner for the recall-first classroom detectors.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--run-speech", action="store_true", help="Also run VSD and VLP smoke steps.")
    parser.add_argument("--seat-calibration", default=None, help="Optional seat calibration JSON for seat-aware smoke tests.")
    parser.add_argument("--topic-config", default="configs/topic_profiles.yaml", help="Topic profile YAML used by the VLP smoke step.")
    parser.add_argument("--course-profile", default="default", help="Course profile name used by the VLP smoke step.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    py = sys.executable
    media_dir = PROJECT_ROOT / "media"
    output_dir = PROJECT_ROOT / "outputs" / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    attention_cmd = [
        py,
        "detectors/attention_detector/run.py",
        "--video",
        str(media_dir / "classroom.mp4"),
        "--config",
        "configs/config.yaml",
        "--output-dir",
        str(output_dir / "attention"),
        "--headless",
        "--device",
        args.device,
    ]
    if args.seat_calibration:
        attention_cmd += ["--seat-calibration", args.seat_calibration]
    run_step(attention_cmd, "attention")

    activity_cmd = [
        py,
        "detectors/activity_detector/run.py",
        "--source",
        str(media_dir / "classroom.mp4"),
        "--out",
        str(output_dir / "activity.mp4"),
        "--activity_out",
        str(output_dir / "activity.csv"),
        "--device",
        args.device,
        "--max_frames",
        str(args.max_frames),
    ]
    if args.seat_calibration:
        activity_cmd += ["--seat_calibration", args.seat_calibration]
    run_step(activity_cmd, "activity")

    if args.run_speech:
        run_step(
            [
                py,
                "detectors/vsd_detector/run.py",
                "--input",
                str(media_dir / "talking1.mp4"),
                "--output",
                str(output_dir / "vsd.mp4"),
                "--device",
                args.device,
                "--max-frames",
                str(args.max_frames),
            ],
            "vsd",
        )
        vlp_cmd = [
            py,
            "detectors/vlp_detector/run.py",
            "--input",
            str(media_dir / "talking2.mp4"),
            "--output",
            str(output_dir / "vlp.mp4"),
            "--csv",
            str(output_dir / "speech_topic_segments.csv"),
            "--device",
            args.device,
            "--max-frames",
            str(args.max_frames),
            "--course-profile",
            args.course_profile,
            "--topic-config",
            args.topic_config,
        ]
        if args.seat_calibration:
            vlp_cmd += ["--seat-calibration", args.seat_calibration]
        run_step(vlp_cmd, "vlp")


if __name__ == "__main__":
    main()
