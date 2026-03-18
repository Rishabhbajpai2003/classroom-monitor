from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


def resolve_torch_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preference)


def expand_box(
    bbox: np.ndarray,
    frame_shape: Tuple[int, int, int],
    scale_x: float = 1.35,
    scale_y: float = 1.55,
    shift_y: float = 0.06,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    height, width = frame_shape[:2]

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5 + (y2 - y1) * shift_y
    box_w = (x2 - x1) * scale_x
    box_h = (y2 - y1) * scale_y

    nx1 = max(0, int(round(cx - box_w * 0.5)))
    ny1 = max(0, int(round(cy - box_h * 0.5)))
    nx2 = min(width, int(round(cx + box_w * 0.5)))
    ny2 = min(height, int(round(cy + box_h * 0.5)))
    return nx1, ny1, nx2, ny2


def crop_face_context(frame_bgr: np.ndarray, bbox: np.ndarray, crop_size: int = 96) -> np.ndarray:
    x1, y1, x2, y2 = expand_box(bbox, frame_bgr.shape)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)


def estimate_lip_box(
    bbox: np.ndarray,
    frame_shape: Tuple[int, int, int],
    width_ratio: float = 0.56,
    height_ratio: float = 0.24,
    center_y_ratio: float = 0.74,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    frame_h, frame_w = frame_shape[:2]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = y1 + center_y_ratio * height
    lip_w = width * width_ratio
    lip_h = height * height_ratio

    lx1 = max(0, int(round(cx - 0.5 * lip_w)))
    ly1 = max(0, int(round(cy - 0.5 * lip_h)))
    lx2 = min(frame_w, int(round(cx + 0.5 * lip_w)))
    ly2 = min(frame_h, int(round(cy + 0.5 * lip_h)))
    return lx1, ly1, lx2, ly2


def draw_lip_box(
    frame_bgr: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.22,
    thickness: int = 2,
) -> Tuple[int, int, int, int]:
    lx1, ly1, lx2, ly2 = estimate_lip_box(bbox, frame_bgr.shape)
    if lx2 <= lx1 or ly2 <= ly1:
        return lx1, ly1, lx2, ly2

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0, dst=frame_bgr)
    cv2.rectangle(frame_bgr, (lx1, ly1), (lx2, ly2), color, thickness)
    return lx1, ly1, lx2, ly2


def get_gap_fill_tracks(tracks: Dict[int, object], frame_idx: int, max_gap_frames: int) -> List[object]:
    active_tracks = []
    for track in tracks.values():
        if frame_idx - int(track.last_frame_idx) <= max_gap_frames:
            active_tracks.append(track)
    active_tracks.sort(key=lambda track: int(track.track_id))
    return active_tracks


def temporal_subsample_frames(frames: List[np.ndarray], target_length: int) -> List[np.ndarray]:
    if target_length <= 0 or len(frames) <= target_length:
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, target_length)
    return [frames[int(round(idx))] for idx in indices]


def clip_to_tensor(frames_bgr: Iterable[np.ndarray]) -> torch.Tensor:
    arrays = []
    for frame in frames_bgr:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        arrays.append(np.transpose(rgb, (2, 0, 1)))
    return torch.from_numpy(np.stack(arrays, axis=0))


@dataclass
class TrackClip:
    frames: Deque[np.ndarray] = field(default_factory=deque)
    frame_indices: Deque[int] = field(default_factory=deque)
    last_seen_frame_idx: int = -1


class TrackClipBuffer:
    def __init__(self, clip_len: int, crop_size: int = 96, max_idle_frames: int = 50) -> None:
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.max_idle_frames = max_idle_frames
        self._tracks: Dict[int, TrackClip] = defaultdict(
            lambda: TrackClip(
                frames=deque(maxlen=clip_len),
                frame_indices=deque(maxlen=clip_len),
            )
        )

    def push(self, track_id: int, frame_idx: int, frame_bgr: np.ndarray, bbox: np.ndarray) -> None:
        clip = self._tracks[track_id]
        clip.frames.append(crop_face_context(frame_bgr, bbox, crop_size=self.crop_size))
        clip.frame_indices.append(frame_idx)
        clip.last_seen_frame_idx = frame_idx

    def ready(self, track_id: int, min_frames: Optional[int] = None) -> bool:
        needed = self.clip_len if min_frames is None else min_frames
        clip = self._tracks.get(track_id)
        return clip is not None and len(clip.frames) >= needed

    def get_tensor(self, track_id: int, device: torch.device, min_frames: Optional[int] = None) -> Optional[torch.Tensor]:
        if not self.ready(track_id, min_frames=min_frames):
            return None
        clip = self._tracks[track_id]
        return clip_to_tensor(clip.frames).unsqueeze(0).to(device)

    def get_frames(self, track_id: int) -> List[np.ndarray]:
        clip = self._tracks.get(track_id)
        if clip is None:
            return []
        return list(clip.frames)

    def get_frame_indices(self, track_id: int) -> List[int]:
        clip = self._tracks.get(track_id)
        if clip is None:
            return []
        return list(clip.frame_indices)

    def prune(self, current_frame_idx: int, active_track_ids: Iterable[int]) -> None:
        active = set(active_track_ids)
        to_delete = []
        for track_id, clip in self._tracks.items():
            if track_id in active:
                continue
            if current_frame_idx - clip.last_seen_frame_idx > self.max_idle_frames:
                to_delete.append(track_id)
        for track_id in to_delete:
            del self._tracks[track_id]


def build_sinusoidal_positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-np.log(10000.0) / dim))
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term)
    return encoding.unsqueeze(0)


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x + identity)
        return x


class SpatioTemporalFrontEnd(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, out_channels: int = 128) -> None:
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock2d(base_channels, out_channels, stride=2),
            ResidualBlock2d(out_channels, out_channels),
            ResidualBlock2d(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is [B, T, C, H, W].
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        batch, channels, time, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        x = self.stage2(x)
        _, channels, height, width = x.shape
        x = x.reshape(batch, time, channels, height, width)
        return x


class VisualTransformerPooling(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        d_model: int = 256,
        grid_size: Tuple[int, int] = (24, 24),
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        token_count = grid_size[0] * grid_size[1]

        self.input_proj = nn.Linear(in_channels, d_model)
        self.spatial_pos = nn.Parameter(torch.zeros(1, token_count, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_query = nn.Parameter(torch.randn(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, time, channels, height, width = x.shape
        if (height, width) != self.grid_size:
            raise ValueError(f"Expected grid size {self.grid_size}, got {(height, width)}")

        tokens = x.reshape(batch * time, channels, height * width).transpose(1, 2)
        tokens = self.input_proj(tokens)
        tokens = self.encoder(tokens + self.spatial_pos)
        tokens = self.norm(tokens)

        scores = torch.matmul(tokens, self.attention_query)
        attention = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(tokens * attention, dim=1)
        pooled = pooled.reshape(batch, time, -1)
        attention_maps = attention.reshape(batch, time, height, width)
        return pooled, attention_maps


class VisualSpeechEncoder(nn.Module):
    def __init__(
        self,
        face_size: int = 96,
        cnn_channels: int = 128,
        d_model: int = 256,
        vtp_layers: int = 6,
        temporal_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if face_size % 4 != 0:
            raise ValueError("face_size must be divisible by 4")

        self.front_end = SpatioTemporalFrontEnd(out_channels=cnn_channels)
        self.pooling = VisualTransformerPooling(
            in_channels=cnn_channels,
            d_model=d_model,
            grid_size=(face_size // 4, face_size // 4),
            num_layers=vtp_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.temporal_norm = nn.LayerNorm(d_model)

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        spatial_features = self.front_end(video)
        pooled_frames, attention_maps = self.pooling(spatial_features)
        positional = build_sinusoidal_positional_encoding(
            pooled_frames.shape[1],
            pooled_frames.shape[2],
            pooled_frames.device,
        )
        encoded = self.temporal_encoder(pooled_frames + positional)
        encoded = self.temporal_norm(encoded)
        return encoded, {
            "attention_maps": attention_maps,
            "frame_embeddings": pooled_frames,
        }


class VisualSpeechDetector(nn.Module):
    def __init__(self, encoder: Optional[VisualSpeechEncoder] = None, d_model: int = 256) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else VisualSpeechEncoder(d_model=d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        encoded, aux = self.encoder(video)
        logits = self.classifier(encoded).squeeze(-1)
        return logits, aux

    @torch.no_grad()
    def predict_proba(self, video: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits, aux = self(video)
        return torch.sigmoid(logits), aux


class LipReadingModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder: Optional[VisualSpeechEncoder] = None,
        d_model: int = 256,
        decoder_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else VisualSpeechEncoder(d_model=d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def encode(self, video: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.encoder(video)

    def decode(self, memory: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(decoder_input_ids)
        positional = build_sinusoidal_positional_encoding(
            embedded.shape[1],
            embedded.shape[2],
            embedded.device,
        )
        target = embedded + positional
        mask = torch.full(
            (target.shape[1], target.shape[1]),
            fill_value=float("-inf"),
            device=target.device,
        )
        mask = torch.triu(mask, diagonal=1)
        decoded = self.decoder(target, memory, tgt_mask=mask)
        decoded = self.decoder_norm(decoded)
        return self.output_head(decoded)

    def forward(self, video: torch.Tensor, decoder_input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        memory, aux = self.encode(video)
        return self.decode(memory, decoder_input_ids), aux

    @torch.no_grad()
    def greedy_decode(
        self,
        video: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 64,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        memory, aux = self.encode(video)
        batch = memory.shape[0]
        tokens = torch.full((batch, 1), bos_token_id, device=memory.device, dtype=torch.long)
        finished = torch.zeros(batch, device=memory.device, dtype=torch.bool)

        for _ in range(max_length - 1):
            logits = self.decode(memory, tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            finished |= next_token.squeeze(1).eq(eos_token_id)
            if bool(torch.all(finished)):
                break

        return tokens, aux


class WordPieceTokenizer:
    def __init__(self, pretrained_name: str = "bert-base-uncased") -> None:
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "transformers is required for the lip-reading tokenizer. "
                "Install it with: pip install transformers"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        if self.tokenizer.cls_token_id is None or self.tokenizer.sep_token_id is None:
            raise RuntimeError("The selected tokenizer must expose [CLS] and [SEP] token ids.")

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    @property
    def bos_token_id(self) -> int:
        return int(self.tokenizer.cls_token_id)

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.sep_token_id)

    @property
    def pad_token_id(self) -> int:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            return 0
        return int(pad_id)

    def encode(self, text: str, max_length: Optional[int] = None) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"][0]

    def decode(self, token_ids: Iterable[int]) -> str:
        skip_ids = {self.pad_token_id, self.bos_token_id}
        cleaned = [int(token_id) for token_id in token_ids if int(token_id) not in skip_ids]
        if cleaned and cleaned[-1] == self.eos_token_id:
            cleaned = cleaned[:-1]
        return self.tokenizer.decode(cleaned, skip_special_tokens=True).strip()
