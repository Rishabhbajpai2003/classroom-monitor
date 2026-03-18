from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from detectors.vsd_detector.common import crop_face_context

try:
    from decord import VideoReader
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "decord is required for VSD training. Install it with: pip install decord"
    ) from exc


POSITIVE_LABELS = {
    "1",
    "2",
    "speaking",
    "speaking_audible",
    "speaking-not-audible",
    "speaking_not_audible",
    "talking",
    "audible",
    "inaudible_speaking",
}
NEGATIVE_LABELS = {
    "0",
    "not_speaking",
    "not-speaking",
    "silent",
    "inactive",
    "nonspeaking",
}


@dataclass
class FrameAnnotation:
    timestamp: float
    bbox: np.ndarray
    label: float


@dataclass
class TrackWindow:
    video_path: str
    video_id: str
    entity_id: str
    frames: List[FrameAnnotation]


def _parse_label(value: object) -> float:
    text = str(value).strip().lower()
    if text in POSITIVE_LABELS:
        return 1.0
    if text in NEGATIVE_LABELS:
        return 0.0
    try:
        numeric = int(float(text))
    except Exception:
        return 0.0
    return 1.0 if numeric in (1, 2) else 0.0


def _parse_bbox_from_string(text: str) -> Optional[np.ndarray]:
    cleaned = (
        text.replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace(";", ",")
        .replace(" ", ",")
    )
    parts = [p for p in cleaned.split(",") if p]
    if len(parts) < 4:
        return None
    try:
        values = [float(parts[i]) for i in range(4)]
    except Exception:
        return None
    return np.asarray(values, dtype=np.float32)


def _parse_bbox_from_row(row: Dict[str, str] | Sequence[str]) -> Optional[np.ndarray]:
    if isinstance(row, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(row.keys()):
            try:
                return np.asarray(
                    [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                    dtype=np.float32,
                )
            except Exception:
                return None
        for key in ("entity_box", "bbox", "box"):
            if key in row:
                bbox = _parse_bbox_from_string(str(row[key]))
                if bbox is not None:
                    return bbox
        return None

    if len(row) >= 6:
        try:
            return np.asarray([float(row[2]), float(row[3]), float(row[4]), float(row[5])], dtype=np.float32)
        except Exception:
            pass
    if len(row) >= 3:
        return _parse_bbox_from_string(str(row[2]))
    return None


def _resolve_video_path(videos_root: str | Path, video_id: str, video_index: Dict[str, Path]) -> str:
    if video_id in video_index:
        return str(video_index[video_id])
    raise FileNotFoundError(f"Could not find a video file for video_id '{video_id}' under {videos_root}")


def index_video_paths(videos_root: str | Path) -> Dict[str, Path]:
    root = Path(videos_root)
    if not root.exists():
        raise FileNotFoundError(f"Videos root not found: {root}")

    index: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".mp4", ".mkv", ".avi", ".mov"}:
            continue
        index.setdefault(path.stem, path)
        index.setdefault(path.name, path)
    return index


def load_ava_active_speaker_windows(
    annotation_csv: str | Path,
    videos_root: str | Path,
    clip_len: int,
    clip_stride: int,
    min_track_frames: int = 8,
    max_frame_gap: float = 0.20,
) -> List[TrackWindow]:
    annotation_csv = Path(annotation_csv)
    if not annotation_csv.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {annotation_csv}")

    video_index = index_video_paths(videos_root)
    grouped: Dict[Tuple[str, str], List[FrameAnnotation]] = {}

    with annotation_csv.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = False

        if has_header:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                video_id = str(
                    row.get("video_id")
                    or row.get("video")
                    or row.get("youtube_id")
                    or row.get("clip_id")
                    or ""
                ).strip()
                entity_id = str(row.get("entity_id") or row.get("person_id") or row.get("track_id") or "").strip()
                if not video_id or not entity_id:
                    continue
                bbox = _parse_bbox_from_row(row)
                if bbox is None:
                    continue
                timestamp_text = row.get("frame_timestamp") or row.get("timestamp") or row.get("time")
                if timestamp_text is None:
                    continue
                grouped.setdefault((video_id, entity_id), []).append(
                    FrameAnnotation(
                        timestamp=float(timestamp_text),
                        bbox=bbox,
                        label=_parse_label(row.get("label") or row.get("label_id") or 0),
                    )
                )
        else:
            reader = csv.reader(handle)
            for row in reader:
                if not row or len(row) < 8:
                    continue
                video_id = str(row[0]).strip()
                entity_id = str(row[7]).strip()
                bbox = _parse_bbox_from_row(row)
                if not video_id or not entity_id or bbox is None:
                    continue
                grouped.setdefault((video_id, entity_id), []).append(
                    FrameAnnotation(
                        timestamp=float(row[1]),
                        bbox=bbox,
                        label=_parse_label(row[6]),
                    )
                )

    windows: List[TrackWindow] = []
    for (video_id, entity_id), frames in grouped.items():
        frames.sort(key=lambda item: item.timestamp)
        segments: List[List[FrameAnnotation]] = []
        current: List[FrameAnnotation] = []
        previous_time: Optional[float] = None
        for frame in frames:
            if previous_time is not None and frame.timestamp - previous_time > max_frame_gap:
                if len(current) >= min_track_frames:
                    segments.append(current)
                current = []
            current.append(frame)
            previous_time = frame.timestamp
        if len(current) >= min_track_frames:
            segments.append(current)

        if not segments:
            continue

        video_path = _resolve_video_path(videos_root, video_id, video_index)
        for segment in segments:
            if len(segment) <= clip_len:
                windows.append(TrackWindow(video_path=video_path, video_id=video_id, entity_id=entity_id, frames=segment))
                continue

            stride = max(1, clip_stride)
            for start in range(0, len(segment) - clip_len + 1, stride):
                window = segment[start:start + clip_len]
                windows.append(TrackWindow(video_path=video_path, video_id=video_id, entity_id=entity_id, frames=window))

            tail = segment[-clip_len:]
            if not windows or windows[-1].frames != tail:
                windows.append(TrackWindow(video_path=video_path, video_id=video_id, entity_id=entity_id, frames=tail))

    return windows


_VIDEO_READER_CACHE: Dict[str, VideoReader] = {}


def _get_video_reader(video_path: str) -> VideoReader:
    reader = _VIDEO_READER_CACHE.get(video_path)
    if reader is None:
        reader = VideoReader(video_path)
        _VIDEO_READER_CACHE[video_path] = reader
    return reader


def _to_absolute_bbox(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
    if float(np.max(bbox)) <= 1.5:
        x1, y1, x2, y2 = bbox
        return np.asarray([x1 * width, y1 * height, x2 * width, y2 * height], dtype=np.float32)
    return bbox.astype(np.float32)


def _apply_train_aug(crop_bgr: np.ndarray, frame_size: int, img_size: int, max_rotation: float, crop_jitter: int) -> np.ndarray:
    working = crop_bgr
    if max_rotation > 0:
        angle = random.uniform(-max_rotation, max_rotation)
        center = (frame_size * 0.5, frame_size * 0.5)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        working = cv2.warpAffine(working, matrix, (frame_size, frame_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    max_shift = max(0, crop_jitter)
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    base = (frame_size - img_size) // 2
    left = int(np.clip(base + shift_x, 0, frame_size - img_size))
    top = int(np.clip(base + shift_y, 0, frame_size - img_size))
    working = working[top:top + img_size, left:left + img_size]
    if random.random() < 0.5:
        working = cv2.flip(working, 1)
    return working


def _apply_eval_crop(crop_bgr: np.ndarray, frame_size: int, img_size: int) -> np.ndarray:
    base = (frame_size - img_size) // 2
    return crop_bgr[base:base + img_size, base:base + img_size]


class AVAActiveSpeakerDataset(Dataset):
    def __init__(
        self,
        annotation_csv: str | Path,
        videos_root: str | Path,
        clip_len: int = 25,
        clip_stride: int = 12,
        frame_size: int = 160,
        img_size: int = 96,
        train: bool = True,
        max_rotation: float = 10.0,
        crop_jitter: int = 3,
        min_track_frames: int = 8,
    ) -> None:
        self.samples = load_ava_active_speaker_windows(
            annotation_csv=annotation_csv,
            videos_root=videos_root,
            clip_len=clip_len,
            clip_stride=clip_stride,
            min_track_frames=min_track_frames,
        )
        self.frame_size = frame_size
        self.img_size = img_size
        self.train = train
        self.max_rotation = max_rotation
        self.crop_jitter = crop_jitter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        reader = _get_video_reader(sample.video_path)
        fps = float(reader.get_avg_fps())
        if fps <= 0:
            fps = 25.0

        frame_indices = [max(0, min(len(reader) - 1, int(round(item.timestamp * fps)))) for item in sample.frames]
        batch = reader.get_batch(frame_indices).asnumpy()  # RGB uint8

        processed_frames: List[np.ndarray] = []
        labels: List[float] = []
        for raw_rgb, annotation in zip(batch, sample.frames):
            frame_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
            height, width = frame_bgr.shape[:2]
            bbox_abs = _to_absolute_bbox(annotation.bbox, width, height)
            crop = crop_face_context(frame_bgr, bbox_abs, crop_size=self.frame_size)
            if self.train:
                crop = _apply_train_aug(crop, self.frame_size, self.img_size, self.max_rotation, self.crop_jitter)
            else:
                crop = _apply_eval_crop(crop, self.frame_size, self.img_size)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            processed_frames.append(np.transpose(rgb, (2, 0, 1)))
            labels.append(float(annotation.label))

        video = torch.from_numpy(np.stack(processed_frames, axis=1)).float()  # [C, T, H, W]
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        mask = torch.ones(len(labels), dtype=torch.bool)

        return {
            "video": video,
            "labels": label_tensor,
            "mask": mask,
            "video_id": sample.video_id,
            "entity_id": sample.entity_id,
            "video_path": sample.video_path,
        }


def collate_vsd_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    max_len = max(int(item["video"].shape[1]) for item in batch)
    height = int(batch[0]["video"].shape[2])
    width = int(batch[0]["video"].shape[3])
    batch_size = len(batch)

    videos = torch.zeros((batch_size, 3, max_len, height, width), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_len), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    video_ids: List[str] = []
    entity_ids: List[str] = []

    for idx, item in enumerate(batch):
        video = item["video"]
        length = int(video.shape[1])
        videos[idx, :, :length] = video
        labels[idx, :length] = item["labels"]
        mask[idx, :length] = item["mask"]
        video_ids.append(str(item["video_id"]))
        entity_ids.append(str(item["entity_id"]))

    return {
        "video": videos,
        "labels": labels,
        "mask": mask,
        "video_ids": video_ids,
        "entity_ids": entity_ids,
    }
