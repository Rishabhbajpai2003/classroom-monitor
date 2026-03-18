import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run both Attention and Activity Detectors")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--device", default=None, help="Compute device (e.g. 0 or cpu)")
    return parser.parse_args()


def run_command(cmd_list, description):
    print(f"\n{'='*60}")
    print(f"  Starting: {description}")
    print(f"{'='*60}")
    
    try:
        # We use strict check=True so pipeline fails surface immediately
        subprocess.run(cmd_list, check=True)
        print(f"✅ Successfully completed {description}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}: {e}")
        sys.exit(1)


def main():
    args = parse_args()
    source = args.source

    python_exe = sys.executable

    # Command 1: Attention Detector
    cmd_attention = [
        python_exe,
        "detectors/attention_detector/run.py",
        "--video", source,
        "--config", "configs/config.yaml",
        "--output-dir", "outputs"
    ]
    if args.device:
        cmd_attention.extend(["--device", args.device])

    # Command 2: Activity Detector
    cmd_activity = [
        python_exe,
        "detectors/activity_detector/run.py",
        "--source", source,
        "--out", "outputs/activity_tracking.mp4",
        "--activity_out", "outputs/person_activity_summary.csv"
    ]
    if args.device:
        cmd_activity.extend(["--device", args.device])

    print(f"Running Unified Classroom Monitor Pipeline on source: {source}")
    
    # 1. Run Attention Pipeline
    run_command(cmd_attention, "Attention Detector Pipeline")
    
    # 2. Run Activity Pipeline
    run_command(cmd_activity, "Activity Detector Pipeline")

    print("\n🎉 All detection pipelines completed successfully! Check the 'outputs/' folder.")


if __name__ == "__main__":
    main()
