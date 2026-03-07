from pathlib import Path
import shutil
from urllib.parse import urlparse
import uuid


def is_url(text):
    try:
        p = urlparse(str(text).strip())
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False


def is_youtube_url(text):
    if not is_url(text):
        return False
    host = (urlparse(str(text)).netloc or "").lower()
    return any(k in host for k in ["youtube.com", "youtu.be", "m.youtube.com"])


def _available_js_runtimes():
    runtimes = {}
    for name in ["node", "deno", "bun"]:
        if shutil.which(name):
            runtimes[name] = {}
    return runtimes


def resolve_source(source, download_dir, keep_downloaded_source=False):
    source_str = str(source).strip()
    source_path = Path(source_str)
    if source_path.exists():
        return str(source_path), None

    if not is_url(source_str):
        raise RuntimeError(f"Source not found as local file and not a URL: {source_str}")

    if not is_youtube_url(source_str):
        # For non-YouTube URLs, let OpenCV try direct streaming.
        return source_str, None

    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError(
            "YouTube URL input requires yt-dlp. Install with: pip install yt-dlp"
        ) from e

    dl_dir = Path(download_dir)
    dl_dir.mkdir(parents=True, exist_ok=True)
    temp_stem = f"yt_{uuid.uuid4().hex}"
    outtmpl = str(dl_dir / f"{temp_stem}.%(ext)s")

    ydl_opts = {
        "format": "best[ext=mp4][vcodec!=none][acodec!=none]/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "noprogress": True,
        "restrictfilenames": True,
        "retries": 3,
        "remote_components": ["ejs:github"],
    }
    js_runtimes = _available_js_runtimes()
    if js_runtimes:
        ydl_opts["js_runtimes"] = js_runtimes

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source_str, download=True)
        downloaded = Path(ydl.prepare_filename(info))

    if not downloaded.exists():
        candidates = sorted(dl_dir.glob(f"{temp_stem}.*"))
        if not candidates:
            raise RuntimeError("Failed to download YouTube video.")
        downloaded = candidates[0]

    cleanup_path = None if keep_downloaded_source else downloaded
    return str(downloaded), cleanup_path
