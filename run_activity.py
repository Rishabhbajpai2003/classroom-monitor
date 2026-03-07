from app.pipeline.activity_config import parse_args
from app.pipeline.activity_processor import run_pipeline


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
