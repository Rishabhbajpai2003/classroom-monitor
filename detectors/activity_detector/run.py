import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from detectors.activity_detector.config import parse_args
from detectors.activity_detector.processor import run_pipeline


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
