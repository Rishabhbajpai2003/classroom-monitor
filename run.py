"""
Entry point for the Classroom Attention + Attendance Monitoring Pipeline.

Usage:
    python run.py --video input.mp4 --camera cam_01
    python run.py --video input.mp4 --camera cam_01 --config config.yaml
    python run.py --video input.mp4 --headless
"""

import argparse
import logging
import os
import sys

import yaml

from app.pipeline.processor import ClassroomProcessor


def setup_logging(level: str = "INFO"):
    """Configure structured logging for the pipeline."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode="a"),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load pipeline configuration from YAML file."""
    if not os.path.exists(config_path):
        logging.warning(
            f"Config file not found: {config_path}. Using defaults."
        )
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    logging.info(f"Loaded config from: {config_path}")
    return config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classroom Attendance + Attention Monitoring Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py --video input.mp4
    python run.py --video input.mp4 --camera cam_02
    python run.py --video media/input.mp4 --config configs/config.yaml --headless
    python run.py --video media/input.mp4 --output-dir outputs/
        """,
    )

    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--camera", "-c",
        type=str,
        default=None,
        help="Camera ID (e.g., cam_01). Overrides config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for video and CSV (overrides config).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode (no display windows).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device (overrides config).",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Ensure 'system' section exists
    if "system" not in config:
        config["system"] = {}

    # CLI overrides
    if args.camera:
        config["system"]["camera_id"] = args.camera
    if args.output_dir:
        config["system"]["output_dir"] = args.output_dir
    if args.headless:
        config["system"]["headless"] = True
    if args.device:
        config["system"]["device"] = args.device

    # Setup logging
    log_level = config.get("system", {}).get("log_level", "INFO")
    setup_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  Classroom Monitoring Pipeline Starting")
    logger.info("=" * 60)
    logger.info(f"  Video  : {args.video}")
    logger.info(f"  Camera : {config['system'].get('camera_id', 'cam_01')}")
    logger.info(f"  Config : {args.config}")
    logger.info(f"  Device : {config['system'].get('device', 'auto')}")

    # Validate input video
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        print(f"\n❌ Error: Video file not found: {args.video}")
        sys.exit(1)

    # Create output directory
    output_dir = config["system"].get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize and run processor
    try:
        processor = ClassroomProcessor(config)
        output_video = os.path.join(output_dir, "output.avi")
        processor.process_video(
            video_path=args.video,
            output_path=output_video,
        )
        logger.info("Pipeline completed successfully.")
        print("\n✅ Pipeline completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        print(f"\n❌ File error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        print("\n⚠️  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()