import numpy as np


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def center_xy(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def point_to_box_distance(point_xy, box_xyxy):
    px, py = float(point_xy[0]), float(point_xy[1])
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx = clamp(px, x1, x2)
    cy = clamp(py, y1, y2)
    return float(np.hypot(px - cx, py - cy))


def box_to_box_distance(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float(np.hypot(dx, dy))


def bbox_from_landmarks(landmarks_xy, w, h, pad=10):
    xs = [p[0] for p in landmarks_xy]
    ys = [p[1] for p in landmarks_xy]
    x1 = clamp(int(min(xs) * w) - pad, 0, w - 1)
    y1 = clamp(int(min(ys) * h) - pad, 0, h - 1)
    x2 = clamp(int(max(xs) * w) + pad, 0, w - 1)
    y2 = clamp(int(max(ys) * h) + pad, 0, h - 1)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def normalize_list(csv_str):
    return [x.strip().lower() for x in csv_str.split(",") if x.strip()]
