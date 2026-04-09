from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_TOPIC_CONFIG = PROJECT_ROOT / "configs" / "topic_profiles.yaml"

app = FastAPI(title="Classroom Monitor API")


def _convert_avi_to_mp4(avi_path: Path) -> Path:
    if not avi_path.exists():
        return avi_path

    mp4_path = avi_path.with_suffix(".mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(avi_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ],
            capture_output=True,
            check=True,
        )
        if mp4_path.exists() and mp4_path.stat().st_size > 0:
            return mp4_path
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(str(avi_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        if mp4_path.exists() and mp4_path.stat().st_size > 0:
            return mp4_path
    except Exception:
        pass

    return avi_path


def _run_subprocess(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    log_text = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return result.returncode, log_text


def _save_optional_upload(upload: UploadFile | None, target_dir: Path, fallback_name: str) -> Path | None:
    if upload is None:
        return None
    suffix = Path(upload.filename or fallback_name).suffix or Path(fallback_name).suffix
    path = target_dir / f"{Path(fallback_name).stem}{suffix}"
    with path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return path


def _path_or_none(path: Path | None) -> str | None:
    return None if path is None or not path.exists() else str(path)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    device: str = Form("auto"),
    camera_id: str = Form("cam_01"),
    run_attention: bool = Form(True),
    run_activity: bool = Form(True),
    run_speech: bool = Form(False),
    course_profile: str = Form("default"),
    seat_calibration: UploadFile | None = File(None),
    topic_config: UploadFile | None = File(None),
) -> dict:
    if not run_attention and not run_activity and not run_speech:
        raise HTTPException(status_code=400, detail="At least one detector must be enabled.")

    run_id = uuid.uuid4().hex[:8]
    run_dir = OUTPUTS_ROOT / f"api_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    input_path = run_dir / f"input{suffix}"
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    seat_calibration_path = _save_optional_upload(seat_calibration, run_dir, "seat_map_calibration.json")
    topic_config_path = _save_optional_upload(topic_config, run_dir, "topic_profiles.yaml") or DEFAULT_TOPIC_CONFIG

    logs: list[str] = [
        f"Run ID: {run_id}",
        f"Output dir: {run_dir}",
        f"Device: {device}",
        f"Camera ID: {camera_id}",
        f"Course profile: {course_profile}",
    ]

    response: dict[str, object] = {
        "ok": True,
        "run_id": run_id,
        "output_dir": str(run_dir),
        "attention_video": None,
        "attendance_csv": None,
        "activity_video": None,
        "activity_csv": None,
        "speech_video": None,
        "speech_topics_csv": None,
        "seat_map_json": None,
        "seat_map_png": None,
        "student_seating_timeline_csv": None,
        "attendance_events_csv": None,
        "logs": logs,
    }

    if run_attention:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "detectors" / "attention_detector" / "run.py"),
            "--video",
            str(input_path),
            "--config",
            str(PROJECT_ROOT / "configs" / "config.yaml"),
            "--output-dir",
            str(run_dir),
            "--headless",
        ]
        if device != "auto":
            cmd += ["--device", device]
        if camera_id.strip():
            cmd += ["--camera", camera_id.strip()]
        if seat_calibration_path is not None:
            cmd += ["--seat-calibration", str(seat_calibration_path)]

        return_code, log_text = _run_subprocess(cmd)
        if log_text:
            logs.append(log_text)
        if return_code != 0:
            raise HTTPException(
                status_code=500,
                detail={"message": "Attention detector failed.", "run_id": run_id, "logs": logs},
            )

        attn_avi = run_dir / "output.avi"
        attn_csv = run_dir / "attendance_report.csv"
        response["attention_video"] = _path_or_none(_convert_avi_to_mp4(attn_avi) if attn_avi.exists() else None)
        response["attendance_csv"] = _path_or_none(attn_csv)

    if run_activity:
        act_video = run_dir / "activity_tracking.mp4"
        act_csv = run_dir / "person_activity_summary.csv"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "detectors" / "activity_detector" / "run.py"),
            "--source",
            str(input_path),
            "--out",
            str(act_video),
            "--activity_out",
            str(act_csv),
        ]
        if device != "auto":
            cmd += ["--device", device]
        if seat_calibration_path is not None:
            cmd += ["--seat_calibration", str(seat_calibration_path)]

        return_code, log_text = _run_subprocess(cmd)
        if log_text:
            logs.append(log_text)
        if return_code != 0:
            raise HTTPException(
                status_code=500,
                detail={"message": "Activity detector failed.", "run_id": run_id, "logs": logs},
            )

        response["activity_video"] = _path_or_none(act_video if act_video.exists() and act_video.stat().st_size > 0 else None)
        response["activity_csv"] = _path_or_none(act_csv)

    if run_speech:
        speech_video = run_dir / "speech_topics.mp4"
        speech_csv = run_dir / "speech_topic_segments.csv"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "detectors" / "vlp_detector" / "run.py"),
            "--input",
            str(input_path),
            "--output",
            str(speech_video),
            "--csv",
            str(speech_csv),
            "--course-profile",
            course_profile,
            "--topic-config",
            str(topic_config_path),
        ]
        if device != "auto":
            cmd += ["--device", device]
        if seat_calibration_path is not None:
            cmd += ["--seat-calibration", str(seat_calibration_path)]

        return_code, log_text = _run_subprocess(cmd)
        if log_text:
            logs.append(log_text)
        if return_code != 0:
            raise HTTPException(
                status_code=500,
                detail={"message": "Speech detector failed.", "run_id": run_id, "logs": logs},
            )

        response["speech_video"] = _path_or_none(speech_video if speech_video.exists() and speech_video.stat().st_size > 0 else None)
        response["speech_topics_csv"] = _path_or_none(speech_csv)

    response["seat_map_json"] = _path_or_none(run_dir / "seat_map.json")
    response["seat_map_png"] = _path_or_none(run_dir / "seat_map.png")
    response["student_seating_timeline_csv"] = _path_or_none(run_dir / "student_seating_timeline.csv")
    response["attendance_events_csv"] = _path_or_none(run_dir / "attendance_events.csv")
    return response
