"""
Classroom Monitor — Gradio Web Interface & API
----------------------------------------------
Exposes both detection pipelines as a web app and REST API.
GPU inference runs locally on this machine; Cloudflare tunnels
requests in from the public internet.

Start the app:
    python gradio_app.py

Expose publicly via Cloudflare (laptop GPU, public URL):
    # In a second terminal:
    cloudflared tunnel --url http://localhost:7860
    # Or run the helper script:
    powershell -ExecutionPolicy Bypass -File launch_cloudflare.ps1
"""

import gradio as gr
import os
import shutil
import subprocess
import sys
import uuid

import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _convert_avi_to_mp4(avi_path: str) -> str:
    """
    Convert an AVI file to MP4 for browser playback.
    Tries ffmpeg first, then falls back to OpenCV.
    Returns the MP4 path on success, original path on failure.
    """
    if not avi_path or not os.path.exists(avi_path):
        return avi_path

    mp4_path = avi_path[:-4] + ".mp4"

    # Try ffmpeg (fastest, best quality)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", avi_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart", mp4_path,
            ],
            capture_output=True,
            check=True,
        )
        if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
            return mp4_path
    except Exception:
        pass

    # Fallback: OpenCV frame-by-frame copy
    try:
        cap = cv2.VideoCapture(avi_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
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


# ─── Core pipeline function ────────────────────────────────────────────────────

def run_classroom_pipeline(
    video_file,
    device: str,
    camera_id: str,
    run_attention: bool,
    run_activity: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Analyze a classroom video for attention, attendance, and activity.

    Parameters
    ----------
    video_file : str
        Path to the uploaded video file (provided by Gradio).
    device : str
        Compute device — ``'auto'``, ``'cuda'``, ``'cpu'``, or ``'mps'``.
    camera_id : str
        Camera / session identifier (e.g. ``'cam_01'``).
    run_attention : bool
        Whether to run the attention & attendance detector.
    run_activity : bool
        Whether to run the activity detector.

    Returns
    -------
    tuple
        ``(attention_video_path, attendance_csv_path,
           activity_video_path, activity_csv_path, log_text)``
    """
    if video_file is None:
        return None, None, None, None, "❌  No video file uploaded."

    # Unique output directory per request
    run_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join(OUTPUTS_ROOT, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Copy uploaded file to a stable path
    ext = os.path.splitext(video_file)[1] or ".mp4"
    input_path = os.path.join(output_dir, f"input{ext}")
    shutil.copy(video_file, input_path)

    py = sys.executable
    logs = [
        f"▶  Run ID     : {run_id}",
        f"▶  Output dir : {output_dir}",
        f"▶  Device     : {device}",
        f"▶  Camera ID  : {camera_id}",
        "",
    ]

    attn_video: str | None = None
    attn_csv: str | None = None
    act_video: str | None = None
    act_csv: str | None = None

    # ── 1. Attention & Attendance Detector ────────────────────────────────────
    if run_attention:
        progress(0.05, desc="Running attention & attendance detector…")
        logs += ["=" * 56, "  🧠  Attention & Attendance Detector", "=" * 56]

        cmd = [
            py,
            os.path.join(
                PROJECT_ROOT, "detectors", "attention_detector", "run.py"
            ),
            "--video", input_path,
            "--config", os.path.join(PROJECT_ROOT, "configs", "config.yaml"),
            "--output-dir", output_dir,
            "--headless",
        ]
        if device != "auto":
            cmd += ["--device", device]
        if camera_id.strip():
            cmd += ["--camera", camera_id.strip()]

        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if res.stdout.strip():
                logs.append(res.stdout.strip())
            if res.returncode != 0:
                logs.append(f"⚠️  stderr:\n{res.stderr.strip()}")
            else:
                logs.append("✅  Attention detector finished successfully.")
        except Exception as exc:
            logs.append(f"❌  Failed to start attention detector: {exc}")

        _avi = os.path.join(output_dir, "output.avi")
        _csv = os.path.join(output_dir, "attendance_report.csv")
        if os.path.exists(_avi):
            attn_video = _convert_avi_to_mp4(_avi)
        if os.path.exists(_csv):
            attn_csv = _csv

    # ── 2. Activity Detector ──────────────────────────────────────────────────
    if run_activity:
        progress(0.55, desc="Running activity detector…")
        logs += ["", "=" * 56, "  📊  Activity Detector", "=" * 56]

        act_vid_path = os.path.join(output_dir, "activity_tracking.mp4")
        act_csv_path = os.path.join(output_dir, "person_activity_summary.csv")

        cmd = [
            py,
            os.path.join(
                PROJECT_ROOT, "detectors", "activity_detector", "run.py"
            ),
            "--source", input_path,
            "--out", act_vid_path,
            "--activity_out", act_csv_path,
        ]
        if device != "auto":
            cmd += ["--device", device]

        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if res.stdout.strip():
                logs.append(res.stdout.strip())
            if res.returncode != 0:
                logs.append(f"⚠️  stderr:\n{res.stderr.strip()}")
            else:
                logs.append("✅  Activity detector finished successfully.")
        except Exception as exc:
            logs.append(f"❌  Failed to start activity detector: {exc}")

        if os.path.exists(act_vid_path) and os.path.getsize(act_vid_path) > 0:
            act_video = act_vid_path
        if os.path.exists(act_csv_path):
            act_csv = act_csv_path

    progress(1.0, desc="Complete!")
    logs += ["", f"🎉  Pipeline complete!  (run_{run_id})"]
    return attn_video, attn_csv, act_video, act_csv, "\n".join(logs)


# ─── Gradio UI ────────────────────────────────────────────────────────────────

_CSS = """
#run-btn { font-size: 1.05rem; min-height: 48px; }
"""

with gr.Blocks(title="Classroom Monitor") as demo:

    gr.Markdown(
        """
# 🎓 Classroom Monitor
**Attendance · Attention · Activity analysis powered by YOLO + DeepSORT + MediaPipe**

Upload a classroom video and click **▶ Run Analysis**.
Results are returned as downloadable files and are also accessible
via the REST API at `/api/analyze`.
"""
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            video_in = gr.Video(
                label="📹 Upload Classroom Video",
                sources=["upload"],
            )

        with gr.Column(scale=2):
            device_dd = gr.Dropdown(
                choices=["auto", "cuda", "cpu", "mps"],
                value="auto",
                label="🖥️ Compute Device",
                info="'auto' = system decides; 'cuda' = force GPU.",
            )
            cam_id_in = gr.Textbox(
                value="cam_01",
                label="📷 Camera ID",
                placeholder="cam_01",
                info="Identifier stored in the attendance report.",
            )
            run_attn_cb = gr.Checkbox(
                value=True,
                label="Run Attention & Attendance Detector",
            )
            run_act_cb = gr.Checkbox(
                value=True,
                label="Run Activity Detector",
            )
            run_btn = gr.Button(
                "▶ Run Analysis", variant="primary", elem_id="run-btn"
            )

    gr.Markdown("---")
    gr.Markdown("### 📤 Results")

    with gr.Row():
        attn_vid_out = gr.Video(label="🧠 Attention-Annotated Video")
        act_vid_out = gr.Video(label="📊 Activity-Tracked Video")

    with gr.Row():
        attn_csv_out = gr.File(label="📄 Attendance Report (CSV)")
        act_csv_out = gr.File(label="📄 Activity Summary (CSV)")

    log_out = gr.Textbox(
        label="📋 Pipeline Log",
        lines=16,
        interactive=False,
    )

    run_btn.click(
        fn=run_classroom_pipeline,
        inputs=[video_in, device_dd, cam_id_in, run_attn_cb, run_act_cb],
        outputs=[attn_vid_out, attn_csv_out, act_vid_out, act_csv_out, log_out],
        api_name="analyze",
    )

    gr.Markdown(
        """
---
### 🔌 Programmatic API

The named endpoint **`/api/analyze`** accepts all the same inputs and returns file paths.

**Python — `gradio_client`:**
```python
from gradio_client import Client

client = Client("http://localhost:7860")   # or your Cloudflare URL
result = client.predict(
    video_file="classroom.mp4",
    device="cuda",
    camera_id="cam_01",
    run_attention=True,
    run_activity=True,
    api_name="/analyze",
)
# result → (attn_video_path, attendance_csv, activity_video_path, activity_csv, log)
```

**cURL:**
```bash
curl -X POST http://localhost:7860/api/analyze \\
  -F 'data=["path/to/video.mp4","cuda","cam_01",true,true]'
```

---
### 🌐 Cloudflare Tunnel (public HTTPS URL, laptop GPU)

```powershell
# Terminal 1 — start the Gradio app
python gradio_app.py

# Terminal 2 — expose it publicly (requires cloudflared)
cloudflared tunnel --url http://localhost:7860

# Or use the helper script that downloads cloudflared automatically:
powershell -ExecutionPolicy Bypass -File launch_cloudflare.ps1
```
The tunnel prints a URL like `https://xxxx.trycloudflare.com`.
All inference still runs on **this machine's GPU** — Cloudflare only
forwards HTTP traffic.
"""
    )


# ─── Launch ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUTS_ROOT, exist_ok=True)
    demo.launch(
        server_name="0.0.0.0",   # reachable on LAN and via tunnel
        server_port=7860,
        theme=gr.themes.Soft(),
        css=_CSS,
    )
