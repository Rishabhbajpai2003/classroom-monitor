"""
Classroom Monitor - Gradio web interface and API wrapper.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np

from app.models.seat_map import RowGuide, SeatCalibration, load_seat_calibration, save_seat_calibration
from app.models.speech_topics import load_topic_profiles

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CALIBRATION_ROOT = OUTPUTS_ROOT / "calibrations"
DEFAULT_TOPIC_CONFIG = PROJECT_ROOT / "configs" / "topic_profiles.yaml"


def _convert_avi_to_mp4(avi_path: str) -> str:
    if not avi_path or not os.path.exists(avi_path):
        return avi_path

    mp4_path = avi_path[:-4] + ".mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                avi_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                mp4_path,
            ],
            capture_output=True,
            check=True,
        )
        if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
            return mp4_path
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(avi_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
            return mp4_path
    except Exception:
        pass

    return avi_path


def _default_calibration_state(frame_index: int = 0, frame_width: int = 0, frame_height: int = 0) -> dict[str, Any]:
    return {
        "reference_frame_index": int(frame_index),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "front_edge": [],
        "rows": [],
        "exit_polygon": [],
        "pending_row_left": None,
        "pending_row_right": None,
    }


def _extract_reference_frame(video_path: str | None, frame_index: int) -> tuple[Any, str]:
    if not video_path:
        return None, "Upload a classroom video first."

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "Could not open the selected video."
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, "Could not read the requested reference frame."
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb, f"Loaded reference frame {int(frame_index)}."


def _state_from_calibration(calibration: SeatCalibration) -> dict[str, Any]:
    return {
        "reference_frame_index": int(calibration.reference_frame_index),
        "frame_width": int(calibration.frame_width),
        "frame_height": int(calibration.frame_height),
        "front_edge": [list(point) for point in calibration.front_edge],
        "rows": [
            {
                "row_id": row.row_id,
                "left": list(row.left),
                "right": list(row.right),
                "seat_count": int(row.seat_count),
            }
            for row in calibration.rows
        ],
        "exit_polygon": [list(point) for point in calibration.exit_polygon],
        "pending_row_left": None,
        "pending_row_right": None,
    }


def _draw_calibration_preview(reference_frame_rgb, state: dict[str, Any]):
    if reference_frame_rgb is None:
        return None
    frame = cv2.cvtColor(reference_frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    front_edge = state.get("front_edge", [])
    if len(front_edge) == 2:
        p1 = tuple(int(round(v)) for v in front_edge[0])
        p2 = tuple(int(round(v)) for v in front_edge[1])
        cv2.line(frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "FRONT", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        for point in front_edge:
            center = tuple(int(round(v)) for v in point)
            cv2.circle(frame, center, 5, (0, 255, 255), -1, cv2.LINE_AA)

    for row in state.get("rows", []):
        left = tuple(int(round(v)) for v in row["left"])
        right = tuple(int(round(v)) for v in row["right"])
        cv2.line(frame, left, right, (255, 180, 0), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{row['row_id']} ({row['seat_count']})",
            left,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

    for name, color in [("pending_row_left", (0, 255, 0)), ("pending_row_right", (255, 0, 0))]:
        point = state.get(name)
        if point is not None:
            center = tuple(int(round(v)) for v in point)
            cv2.circle(frame, center, 6, color, -1, cv2.LINE_AA)
            cv2.putText(frame, name.replace("_", " "), (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    exit_polygon = state.get("exit_polygon", [])
    if len(exit_polygon) >= 2:
        pts = []
        for point in exit_polygon:
            center = tuple(int(round(v)) for v in point)
            pts.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
        if len(pts) >= 3:
            polygon = cv2.convexHull(np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2)))
            cv2.polylines(frame, [polygon], True, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.polylines(frame, [np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))], False, (0, 0, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _calibration_status(state: dict[str, Any]) -> str:
    return (
        f"Front edge points: {len(state.get('front_edge', []))}/2 | "
        f"Rows: {len(state.get('rows', []))} | "
        f"Exit points: {len(state.get('exit_polygon', []))}"
    )


def load_reference_frame_for_ui(video_path, frame_index, calibration_path):
    reference_frame, message = _extract_reference_frame(video_path, int(frame_index))
    if reference_frame is None:
        return None, None, _default_calibration_state(frame_index), message

    state = _default_calibration_state(frame_index, reference_frame.shape[1], reference_frame.shape[0])
    if calibration_path:
        try:
            calibration = load_seat_calibration(calibration_path)
            state = _state_from_calibration(calibration)
            state["reference_frame_index"] = int(frame_index)
            state["frame_width"] = int(reference_frame.shape[1])
            state["frame_height"] = int(reference_frame.shape[0])
            message = f"{message} Loaded existing calibration."
        except Exception as exc:
            message = f"{message} Could not load calibration: {exc}"
    return _draw_calibration_preview(reference_frame, state), reference_frame, state, f"{message} {_calibration_status(state)}"


def update_calibration_click(reference_frame, state, click_mode, evt: gr.SelectData):
    if reference_frame is None:
        return None, state, "Load a reference frame first."
    if evt is None or evt.index is None:
        return _draw_calibration_preview(reference_frame, state), state, _calibration_status(state)

    x, y = int(evt.index[0]), int(evt.index[1])
    point = [float(x), float(y)]
    if click_mode == "front_edge":
        points = list(state.get("front_edge", []))
        if len(points) >= 2:
            points = points[1:]
        points.append(point)
        state["front_edge"] = points
    elif click_mode == "row_left":
        state["pending_row_left"] = point
    elif click_mode == "row_right":
        state["pending_row_right"] = point
    elif click_mode == "exit_polygon":
        polygon = list(state.get("exit_polygon", []))
        polygon.append(point)
        state["exit_polygon"] = polygon

    return _draw_calibration_preview(reference_frame, state), state, _calibration_status(state)


def add_row_to_calibration(reference_frame, state, row_id, seat_count):
    if reference_frame is None:
        return None, state, "Load a reference frame first."
    left = state.get("pending_row_left")
    right = state.get("pending_row_right")
    if left is None or right is None:
        return _draw_calibration_preview(reference_frame, state), state, "Mark both row endpoints before adding the row."
    seat_count = int(seat_count or 0)
    if seat_count <= 0:
        return _draw_calibration_preview(reference_frame, state), state, "Seat count must be greater than zero."

    row_token = str(row_id).strip() or f"R{len(state.get('rows', [])) + 1:02d}"
    rows = [row for row in state.get("rows", []) if row["row_id"] != row_token]
    rows.append(
        {
            "row_id": row_token,
            "left": list(left),
            "right": list(right),
            "seat_count": seat_count,
        }
    )
    state["rows"] = sorted(rows, key=lambda row: row["row_id"])
    state["pending_row_left"] = None
    state["pending_row_right"] = None
    return _draw_calibration_preview(reference_frame, state), state, f"Added row {row_token}. {_calibration_status(state)}"


def clear_exit_polygon(reference_frame, state):
    state["exit_polygon"] = []
    return _draw_calibration_preview(reference_frame, state), state, f"Cleared exit polygon. {_calibration_status(state)}"


def reset_calibration(reference_frame, state):
    new_state = _default_calibration_state(
        state.get("reference_frame_index", 0),
        state.get("frame_width", 0),
        state.get("frame_height", 0),
    )
    return _draw_calibration_preview(reference_frame, new_state), new_state, "Calibration reset."


def save_calibration_artifact(reference_frame, state):
    if reference_frame is None:
        return None, None, "Load a reference frame first."
    if len(state.get("front_edge", [])) != 2:
        return None, None, "Mark exactly two front-edge points before saving."
    if not state.get("rows"):
        return None, None, "Add at least one seating row before saving."

    calibration = SeatCalibration(
        reference_frame_index=int(state.get("reference_frame_index", 0)),
        frame_width=int(state.get("frame_width", reference_frame.shape[1])),
        frame_height=int(state.get("frame_height", reference_frame.shape[0])),
        front_edge=[list(point) for point in state["front_edge"]],
        rows=[
            RowGuide(
                row_id=str(row["row_id"]),
                left=list(row["left"]),
                right=list(row["right"]),
                seat_count=int(row["seat_count"]),
            )
            for row in state["rows"]
        ],
        exit_polygon=[list(point) for point in state.get("exit_polygon", [])],
        calibration_name=f"seat_map_{uuid.uuid4().hex[:8]}",
    )

    CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
    path = CALIBRATION_ROOT / f"{calibration.calibration_name}.json"
    save_seat_calibration(calibration, path)
    return str(path), str(path), f"Saved seat calibration to {path.name}."


def _load_topic_profile_choices() -> list[str]:
    try:
        profiles = load_topic_profiles(DEFAULT_TOPIC_CONFIG)
        return sorted(profiles.keys())
    except Exception:
        return ["default", "math", "science"]


def run_classroom_pipeline(
    video_file,
    device: str,
    camera_id: str,
    run_attention: bool,
    run_activity: bool,
    run_speech: bool,
    seat_calibration_file,
    built_calibration_path,
    course_profile: str,
    topic_config_file,
    progress=gr.Progress(track_tqdm=True),
):
    if video_file is None:
        return (None,) * 10 + ("No video file uploaded.",)

    run_id = uuid.uuid4().hex[:8]
    output_dir = OUTPUTS_ROOT / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(video_file).suffix or ".mp4"
    input_path = output_dir / f"input{ext}"
    shutil.copy(video_file, input_path)

    calibration_path = seat_calibration_file or built_calibration_path
    topic_config_path = topic_config_file or str(DEFAULT_TOPIC_CONFIG)

    py = sys.executable
    logs = [
        f"Run ID     : {run_id}",
        f"Output dir : {output_dir}",
        f"Device     : {device}",
        f"Camera ID  : {camera_id}",
        f"Course     : {course_profile}",
        f"Seat map   : {Path(calibration_path).name if calibration_path else 'not provided'}",
        "",
    ]

    attn_video = None
    attn_csv = None
    act_video = None
    act_csv = None
    speech_video = None
    speech_csv = None

    if run_attention:
        progress(0.10, desc="Running attention, attendance, and seating pipeline...")
        cmd = [
            py,
            str(PROJECT_ROOT / "detectors" / "attention_detector" / "run.py"),
            "--video",
            str(input_path),
            "--config",
            str(PROJECT_ROOT / "configs" / "config.yaml"),
            "--output-dir",
            str(output_dir),
            "--headless",
        ]
        if device != "auto":
            cmd += ["--device", device]
        if camera_id.strip():
            cmd += ["--camera", camera_id.strip()]
        if calibration_path:
            cmd += ["--seat-calibration", str(calibration_path)]

        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if res.stdout.strip():
            logs.append(res.stdout.strip())
        if res.returncode != 0:
            logs.append(f"Attention detector failed with exit code {res.returncode}.")
            if res.stderr.strip():
                logs.append(res.stderr.strip())
        attn_avi = output_dir / "output.avi"
        attn_csv_path = output_dir / "attendance_report.csv"
        if attn_avi.exists():
            attn_video = _convert_avi_to_mp4(str(attn_avi))
        if attn_csv_path.exists():
            attn_csv = str(attn_csv_path)

    if run_activity:
        progress(0.50, desc="Running device-use and note-taking detector...")
        act_video_path = output_dir / "activity_tracking.mp4"
        act_csv_path = output_dir / "person_activity_summary.csv"
        cmd = [
            py,
            str(PROJECT_ROOT / "detectors" / "activity_detector" / "run.py"),
            "--source",
            str(input_path),
            "--out",
            str(act_video_path),
            "--activity_out",
            str(act_csv_path),
        ]
        if device != "auto":
            cmd += ["--device", device]
        if calibration_path:
            cmd += ["--seat_calibration", str(calibration_path)]

        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if res.stdout.strip():
            logs.append(res.stdout.strip())
        if res.returncode != 0:
            logs.append(f"Activity detector failed with exit code {res.returncode}.")
            if res.stderr.strip():
                logs.append(res.stderr.strip())
        if act_video_path.exists() and act_video_path.stat().st_size > 0:
            act_video = str(act_video_path)
        if act_csv_path.exists():
            act_csv = str(act_csv_path)

    if run_speech:
        progress(0.78, desc="Running speech-topic detector...")
        speech_video_path = output_dir / "speech_topics.mp4"
        speech_csv_path = output_dir / "speech_topic_segments.csv"
        cmd = [
            py,
            str(PROJECT_ROOT / "detectors" / "vlp_detector" / "run.py"),
            "--input",
            str(input_path),
            "--output",
            str(speech_video_path),
            "--csv",
            str(speech_csv_path),
            "--course-profile",
            str(course_profile).strip() or "default",
            "--topic-config",
            str(topic_config_path),
        ]
        if device != "auto":
            cmd += ["--device", device]
        if calibration_path:
            cmd += ["--seat-calibration", str(calibration_path)]

        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if res.stdout.strip():
            logs.append(res.stdout.strip())
        if res.returncode != 0:
            logs.append(f"Speech detector failed with exit code {res.returncode}.")
            if res.stderr.strip():
                logs.append(res.stderr.strip())
        if speech_video_path.exists() and speech_video_path.stat().st_size > 0:
            speech_video = str(speech_video_path)
        if speech_csv_path.exists():
            speech_csv = str(speech_csv_path)

    progress(1.0, desc="Complete")
    seat_map_png = str(output_dir / "seat_map.png") if (output_dir / "seat_map.png").exists() else None
    seat_map_json = str(output_dir / "seat_map.json") if (output_dir / "seat_map.json").exists() else None
    seating_timeline = str(output_dir / "student_seating_timeline.csv") if (output_dir / "student_seating_timeline.csv").exists() else None
    attendance_events = str(output_dir / "attendance_events.csv") if (output_dir / "attendance_events.csv").exists() else None

    logs.append("")
    logs.append(f"Pipeline complete (run_{run_id})")
    return (
        attn_video,
        attn_csv,
        act_video,
        act_csv,
        speech_video,
        speech_csv,
        seat_map_png,
        seat_map_json,
        seating_timeline,
        attendance_events,
        "\n".join(logs),
    )


_CSS = """
#run-btn { font-size: 1.05rem; min-height: 48px; }
#calibration-save-btn { min-height: 42px; }
"""

TOPIC_PROFILES = _load_topic_profile_choices()

with gr.Blocks(title="Classroom Monitor") as demo:
    gr.Markdown(
        """
# Classroom Monitor
Recall-first attendance, seating, activity, and speech-topic analysis for wide classroom videos.

Use the calibration panel to create a per-video seat map, then run the detectors. Student IDs stay aligned across seating, attention, activity, and speech outputs.
"""
    )

    reference_frame_state = gr.State(None)
    calibration_state = gr.State(_default_calibration_state())
    built_calibration_state = gr.State(None)

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            video_in = gr.Video(label="Upload Classroom Video", sources=["upload"])
            with gr.Accordion("Seat Calibration Builder", open=False):
                with gr.Row():
                    ref_frame_idx = gr.Number(value=0, precision=0, label="Reference Frame Index")
                    click_mode = gr.Dropdown(
                        choices=["front_edge", "row_left", "row_right", "exit_polygon"],
                        value="front_edge",
                        label="Click Mode",
                    )
                with gr.Row():
                    row_id_box = gr.Textbox(label="Row ID", value="R01")
                    row_seat_count = gr.Number(label="Row Seat Count", value=6, precision=0)
                with gr.Row():
                    load_ref_btn = gr.Button("Load Reference Frame")
                    add_row_btn = gr.Button("Add Row")
                    clear_exit_btn = gr.Button("Clear Exit")
                    reset_cal_btn = gr.Button("Reset")
                    save_cal_btn = gr.Button("Save Calibration", elem_id="calibration-save-btn")
                seat_editor = gr.Image(label="Seat Calibration Editor", interactive=True)
                calibration_status = gr.Textbox(label="Calibration Status", interactive=False, lines=3)
                built_calibration_file = gr.File(label="Built Calibration JSON")

        with gr.Column(scale=2):
            device_dd = gr.Dropdown(choices=["auto", "cuda", "cpu", "mps"], value="auto", label="Compute Device")
            cam_id_in = gr.Textbox(value="cam_01", label="Camera ID")
            run_attn_cb = gr.Checkbox(value=True, label="Run Attention, Attendance, and Seat Events")
            run_act_cb = gr.Checkbox(value=True, label="Run Device Use / Note-Taking Detector")
            run_speech_cb = gr.Checkbox(value=False, label="Run Speech Topic Detector (VSD + Lip Reading)")
            seat_calibration_upload = gr.File(label="Upload Seat Calibration JSON", type="filepath")
            topic_profile_in = gr.Dropdown(choices=TOPIC_PROFILES, value=TOPIC_PROFILES[0], label="Course Profile")
            topic_config_upload = gr.File(label="Optional Topic Config YAML", type="filepath")
            run_btn = gr.Button("Run Analysis", variant="primary", elem_id="run-btn")

    gr.Markdown("### Results")
    with gr.Row():
        attn_vid_out = gr.Video(label="Attention / Seating Video")
        act_vid_out = gr.Video(label="Activity Video")
        speech_vid_out = gr.Video(label="Speech Topic Video")

    with gr.Row():
        attn_csv_out = gr.File(label="Attendance Report CSV")
        act_csv_out = gr.File(label="Activity Timeline CSV")
        speech_csv_out = gr.File(label="Speech Topic Segments CSV")

    with gr.Row():
        seat_map_png_out = gr.Image(label="Seat Map Preview")
        seat_map_json_out = gr.File(label="Seat Map JSON")
        seating_timeline_out = gr.File(label="Student Seating Timeline CSV")
        attendance_events_out = gr.File(label="Attendance Events CSV")

    log_out = gr.Textbox(label="Pipeline Log", lines=16, interactive=False)

    load_ref_btn.click(
        fn=load_reference_frame_for_ui,
        inputs=[video_in, ref_frame_idx, seat_calibration_upload],
        outputs=[seat_editor, reference_frame_state, calibration_state, calibration_status],
    )

    seat_editor.select(
        fn=update_calibration_click,
        inputs=[reference_frame_state, calibration_state, click_mode],
        outputs=[seat_editor, calibration_state, calibration_status],
    )

    add_row_btn.click(
        fn=add_row_to_calibration,
        inputs=[reference_frame_state, calibration_state, row_id_box, row_seat_count],
        outputs=[seat_editor, calibration_state, calibration_status],
    )

    clear_exit_btn.click(
        fn=clear_exit_polygon,
        inputs=[reference_frame_state, calibration_state],
        outputs=[seat_editor, calibration_state, calibration_status],
    )

    reset_cal_btn.click(
        fn=reset_calibration,
        inputs=[reference_frame_state, calibration_state],
        outputs=[seat_editor, calibration_state, calibration_status],
    )

    save_cal_btn.click(
        fn=save_calibration_artifact,
        inputs=[reference_frame_state, calibration_state],
        outputs=[built_calibration_state, built_calibration_file, calibration_status],
    )

    run_btn.click(
        fn=run_classroom_pipeline,
        inputs=[
            video_in,
            device_dd,
            cam_id_in,
            run_attn_cb,
            run_act_cb,
            run_speech_cb,
            seat_calibration_upload,
            built_calibration_state,
            topic_profile_in,
            topic_config_upload,
        ],
        outputs=[
            attn_vid_out,
            attn_csv_out,
            act_vid_out,
            act_csv_out,
            speech_vid_out,
            speech_csv_out,
            seat_map_png_out,
            seat_map_json_out,
            seating_timeline_out,
            attendance_events_out,
            log_out,
        ],
        api_name="analyze",
    )

    gr.Markdown(
        """
### API

The named endpoint `/api/analyze` mirrors the same pipeline options exposed in the UI. Upload a seat calibration JSON if you want seat-aware attendance, seat shifts, late/early events, and seat-linked speech topics.
"""
    )


if __name__ == "__main__":
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), css=_CSS)
