from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_TOPIC_CONFIG = PROJECT_ROOT / "configs" / "topic_profiles.yaml"
RUN_SUMMARY_NAME = "run_summary.json"
PIPELINE_LOG_NAME = "pipeline_log.txt"

app = FastAPI(title="Classroom Monitor API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_JOB_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")
    return text[:48] or "session"


def _safe_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    return lowered in {"1", "true", "yes", "on"}


def _safe_string(value: str | None, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _today_string() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _job_response(job_id: str) -> dict[str, Any]:
    with _JOB_LOCK:
        if job_id not in _JOBS:
            raise HTTPException(status_code=404, detail="Job not found")
        job = dict(_JOBS[job_id])
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "current_step": job["current_step"],
        "error_message": job.get("error_message"),
        "result": job.get("result"),
    }


def _set_job_state(job_id: str, **updates: Any) -> None:
    with _JOB_LOCK:
        if job_id not in _JOBS:
            return
        _JOBS[job_id].update(updates)


def _path_or_none(path: Path | None) -> str | None:
    return None if path is None or not path.exists() else str(path.resolve())


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


def _save_optional_upload(upload: UploadFile | None, target_dir: Path, fallback_name: str) -> Path | None:
    if upload is None:
        return None
    suffix = Path(upload.filename or fallback_name).suffix or Path(fallback_name).suffix
    path = target_dir / f"{Path(fallback_name).stem}{suffix}"
    with path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return path


def _run_subprocess(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    log_text = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return result.returncode, log_text


def _build_result_payload(
    run_id: str,
    run_dir: Path,
    class_date: str,
    subject_name: str,
    topic: str,
    timestamp: str,
    log_text: str,
) -> dict[str, Any]:
    attn_avi = run_dir / "output.avi"
    attention_video = _path_or_none(_convert_avi_to_mp4(attn_avi) if attn_avi.exists() else None)
    attention_csv = _path_or_none(run_dir / "attendance_report.csv")
    activity_video = _path_or_none(run_dir / "activity_tracking.mp4")
    activity_csv = _path_or_none(run_dir / "person_activity_summary.csv")
    speech_video = _path_or_none(run_dir / "speech_topics.mp4")
    speech_csv = _path_or_none(run_dir / "speech_topic_segments.csv")
    seat_map_png = _path_or_none(run_dir / "seat_map.png")
    seat_map_json = _path_or_none(run_dir / "seat_map.json")
    seating_timeline = _path_or_none(run_dir / "student_seating_timeline.csv")
    attendance_events = _path_or_none(run_dir / "attendance_events.csv")
    log_path = run_dir / PIPELINE_LOG_NAME

    if log_text.strip():
        log_path.write_text(log_text, encoding="utf-8")

    return {
        "run_id": run_id,
        "class_date": class_date,
        "subject_name": subject_name,
        "topic": topic,
        "timestamp": timestamp,
        "attentionVideoPath": attention_video,
        "attendanceCsvPath": attention_csv,
        "activityVideoPath": activity_video,
        "activityCsvPath": activity_csv,
        "speechVideoPath": speech_video,
        "speechCsvPath": speech_csv,
        "seatMapPngPath": seat_map_png,
        "seatMapJsonPath": seat_map_json,
        "seatingTimelinePath": seating_timeline,
        "attendanceEventsPath": attendance_events,
        "logText": log_text or "Pipeline complete.",
        "logTextPath": _path_or_none(log_path if log_path.exists() else None),
    }


def _write_run_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / RUN_SUMMARY_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_run_summaries() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not OUTPUTS_ROOT.exists():
        return summaries
    for summary_path in OUTPUTS_ROOT.rglob(RUN_SUMMARY_NAME):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload["_summary_path"] = str(summary_path.resolve())
        summaries.append(payload)
    summaries.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return summaries


def _filter_summaries_by_subject(summaries: list[dict[str, Any]], subject: str | None) -> list[dict[str, Any]]:
    if not subject:
        return summaries
    wanted = subject.strip().lower()
    return [item for item in summaries if str(item.get("subject_name", "")).strip().lower() == wanted]


def _mock_rag_response(query: str) -> str:
    lower_query = query.lower()
    if "cnn" in lower_query or "convolutional" in lower_query:
        return (
            "A Convolutional Neural Network learns spatial patterns through shared filters. "
            "It is commonly used for images because convolution preserves local structure."
        )
    if "backpropagation" in lower_query:
        return (
            "Backpropagation computes gradients of the loss with respect to network weights by "
            "applying the chain rule from output back to input."
        )
    if "overfitting" in lower_query:
        return (
            "Overfitting happens when a model learns training-specific noise instead of general "
            "patterns, so validation performance drops even while training performance improves."
        )
    return (
        "That topic is not fully wired to a real RAG backend yet, but the study assistant endpoint is live. "
        "I can return concise DL/ML guidance while we keep the screen functional."
    )


def _process_analysis_job(
    job_id: str,
    input_path: Path,
    run_dir: Path,
    device: str,
    camera_id: str,
    run_attention: bool,
    run_activity: bool,
    run_speech: bool,
    course_profile: str,
    seat_calibration_path: Path | None,
    topic_config_path: Path,
    class_date: str,
    subject_name: str,
    class_topic: str,
) -> None:
    logs: list[str] = [
        f"Run ID: {job_id}",
        f"Output dir: {run_dir}",
        f"Device: {device}",
        f"Camera ID: {camera_id}",
        f"Subject: {subject_name}",
        f"Topic: {class_topic}",
        f"Class date: {class_date}",
        f"Course profile: {course_profile}",
    ]

    stages: list[tuple[str, list[str]]] = []
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
        stages.append(("attention/attendance", cmd))

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
        stages.append(("activity", cmd))

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
        stages.append(("speech", cmd))

    try:
        _set_job_state(job_id, status="processing", progress=5, current_step="Preparing analysis run")
        total_stages = max(1, len(stages))
        for index, (label, cmd) in enumerate(stages):
            start_progress = 5 + int(index * 80 / total_stages)
            end_progress = 5 + int((index + 1) * 80 / total_stages)
            _set_job_state(job_id, progress=start_progress, current_step=f"Running {label}")
            return_code, log_text = _run_subprocess(cmd)
            if log_text:
                logs.append(f"[{label}]\n{log_text}")
            if return_code != 0:
                _set_job_state(
                    job_id,
                    status="failed",
                    progress=end_progress,
                    current_step=f"{label} failed",
                    error_message=f"{label} detector failed",
                )
                return
            _set_job_state(job_id, progress=end_progress, current_step=f"Completed {label}")

        _set_job_state(job_id, progress=95, current_step="Packaging results")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = _build_result_payload(
            run_id=job_id,
            run_dir=run_dir,
            class_date=class_date,
            subject_name=subject_name,
            topic=class_topic,
            timestamp=timestamp,
            log_text="\n\n".join(logs + ["Pipeline complete"]),
        )
        _write_run_summary(run_dir, payload)
        _set_job_state(job_id, status="completed", progress=100, current_step="Completed", result=payload)
    except Exception as exc:
        logs.append(f"[internal]\n{exc}")
        _set_job_state(
            job_id,
            status="failed",
            progress=100,
            current_step="Failed",
            error_message=str(exc),
        )


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "online", "service": "Classroom Monitor API", "version": "2.0.0"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analysis/jobs")
async def create_analysis_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    device: str = Form("auto"),
    camera_id: str = Form("cam_01"),
    run_attention: bool = Form(True),
    run_activity: bool = Form(True),
    run_speech: bool = Form(False),
    course_profile: str = Form("default"),
    class_date: str = Form(""),
    class_topic: str = Form("General"),
    subject_name: str = Form(""),
    seat_calibration: UploadFile | None = File(None),
    topic_config: UploadFile | None = File(None),
) -> dict[str, Any]:
    if not str(video.content_type or "").startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    run_id = uuid.uuid4().hex[:8]
    safe_date = _safe_string(class_date, _today_string())
    safe_subject = _safe_string(subject_name, class_topic)
    run_dir = OUTPUTS_ROOT / safe_date / f"{_slugify(safe_subject)}_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    input_path = run_dir / f"input{suffix}"
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    seat_calibration_path = _save_optional_upload(seat_calibration, run_dir, "seat_map_calibration.json")
    topic_config_path = _save_optional_upload(topic_config, run_dir, "topic_profiles.yaml") or DEFAULT_TOPIC_CONFIG

    with _JOB_LOCK:
        _JOBS[run_id] = {
            "status": "queued",
            "progress": 0,
            "current_step": "Queued",
            "error_message": None,
            "result": None,
            "run_dir": str(run_dir),
        }

    background_tasks.add_task(
        _process_analysis_job,
        run_id,
        input_path,
        run_dir,
        _safe_string(device, "auto"),
        _safe_string(camera_id, "cam_01"),
        _safe_bool(run_attention, True),
        _safe_bool(run_activity, True),
        _safe_bool(run_speech, False),
        _safe_string(course_profile, "default"),
        seat_calibration_path,
        topic_config_path,
        safe_date,
        safe_subject,
        _safe_string(class_topic, "General"),
    )

    return {"job_id": run_id, "status": "queued", "progress": 0, "current_step": "Queued"}


@app.get("/api/analysis/jobs/{job_id}")
async def get_analysis_job(job_id: str) -> dict[str, Any]:
    return _job_response(job_id)


@app.get("/api/dates")
async def get_available_dates(subject: str | None = Query(default=None)) -> list[str]:
    summaries = _filter_summaries_by_subject(_load_run_summaries(), subject)
    dates = sorted({str(item.get("class_date", "")) for item in summaries if item.get("class_date")}, reverse=True)
    return dates


@app.get("/api/attendance/{date}")
async def get_attendance_runs(date: str, subject: str | None = Query(default=None)) -> list[dict[str, Any]]:
    summaries = _filter_summaries_by_subject(_load_run_summaries(), subject)
    return [item for item in summaries if str(item.get("class_date", "")) == date]


@app.get("/api/files")
async def download_artifact(path: str = Query(...)) -> FileResponse:
    raw_path = Path(path)
    candidate = raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path)
    candidate = candidate.resolve()
    outputs_root = OUTPUTS_ROOT.resolve()
    if outputs_root not in candidate.parents and candidate != outputs_root:
        raise HTTPException(status_code=403, detail="Requested file is outside the outputs directory")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)


@app.post("/api/query-rag")
async def query_rag(query: str = Form(...), course_id: str = Form("global")) -> dict[str, str]:
    _ = course_id
    return {"answer": _mock_rag_response(query)}
