import cv2
import numpy as np

from app.utils.vision_utils import clamp, center_xy, point_to_box_distance

try:
    import face_recognition
except ImportError:
    face_recognition = None

try:
    _cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _cascade = cv2.CascadeClassifier(_cascade_path)
    FACE_CASCADE = _cascade if _cascade is not None and not _cascade.empty() else None
except Exception:
    FACE_CASCADE = None


def extract_head_point(keypoints_xy, keypoints_conf, conf_thr=0.35):
    pts = [keypoints_xy[i] for i in [0, 1, 2, 3, 4] if i < len(keypoints_conf) and keypoints_conf[i] >= conf_thr]
    if not pts:
        return None
    return np.mean(np.asarray(pts, dtype=np.float32), axis=0)


def person_looking_at_box(person, box_xyxy):
    head = person.get("head")
    if head is None:
        return False
    obj_c = center_xy(box_xyxy)
    px1, py1, px2, py2 = person["box"]
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    max_dist = 1.1 * max(pw, ph)
    d = float(np.linalg.norm(obj_c - head))
    in_vertical_band = (py1 - 0.10 * ph) <= obj_c[1] <= (py2 + 0.15 * ph)
    return d <= max_dist and in_vertical_band


def person_close_to_box(person, box_xyxy):
    person_c = center_xy(person["box"])
    obj_c = center_xy(box_xyxy)
    px1, py1, px2, py2 = person["box"]
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    return float(np.linalg.norm(person_c - obj_c)) <= 0.9 * max(pw, ph)


def person_hand_near_box(person, box_xyxy):
    hand_points = person.get("hand_points", [])
    if not hand_points:
        return False
    x1, y1, x2, y2 = box_xyxy
    obj_size = max(1.0, max(x2 - x1, y2 - y1))
    px1, py1, px2, py2 = person["box"]
    person_size = max(1.0, max(px2 - px1, py2 - py1))
    dist_thr = max(22.0, 0.28 * obj_size, 0.08 * person_size)
    for hp in hand_points:
        if point_to_box_distance(hp, box_xyxy) <= dist_thr:
            return True
    return False


def extract_face_embedding(frame_bgr, person_box, face_model="hog"):
    x1, y1, x2, y2 = [int(v) for v in person_box]
    h, w = frame_bgr.shape[:2]
    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    head_y2 = y1 + int(0.55 * (y2 - y1))
    head_y2 = clamp(head_y2, y1 + 1, y2)
    crop = frame_bgr[y1:head_y2, x1:x2]
    if crop.size == 0:
        return None

    if face_recognition is not None:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model=face_model)
        if not locations:
            return None
        encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)
        if not encodings:
            return None

        def area(loc):
            top, right, bottom, left = loc
            return max(0, right - left) * max(0, bottom - top)

        best_idx = int(np.argmax([area(loc) for loc in locations]))
        return np.asarray(encodings[best_idx], dtype=np.float32)

    if FACE_CASCADE is None:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(24, 24),
    )
    if faces is None or len(faces) == 0:
        return None

    fx, fy, fw, fh = max(faces, key=lambda r: int(r[2]) * int(r[3]))
    face_crop = gray[fy:fy + fh, fx:fx + fw]
    if face_crop.size == 0:
        return None
    face_crop = cv2.equalizeHist(face_crop)
    face_crop = cv2.resize(face_crop, (32, 32), interpolation=cv2.INTER_AREA)
    feat = face_crop.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(feat))
    if norm <= 1e-6:
        return None
    feat /= norm
    return feat


def match_face_id(embedding, face_db, face_match_thresh, used_ids, fallback_face_cos_thresh=0.22):
    if embedding is None or not face_db:
        return None
    best_id = None
    best_dist = 1e9
    best_ok = False
    emb_dim = int(embedding.shape[0])
    for rec in face_db:
        pid = rec["id"]
        if pid in used_ids:
            continue
        rec_emb = rec["embedding"]
        if int(rec_emb.shape[0]) != emb_dim:
            continue
        if emb_dim == 128:
            dist = float(np.linalg.norm(embedding - rec_emb))
            dist_ok = dist <= face_match_thresh
        else:
            a = embedding / (float(np.linalg.norm(embedding)) + 1e-6)
            b = rec_emb / (float(np.linalg.norm(rec_emb)) + 1e-6)
            dist = float(1.0 - np.dot(a, b))
            dist_ok = dist <= fallback_face_cos_thresh
        if dist < best_dist:
            best_dist = dist
            best_id = pid
            best_ok = dist_ok
    if best_id is not None and best_ok:
        return best_id
    return None


def choose_primary_activity(activity_set):
    if any(a.startswith("note-taking") for a in activity_set):
        return "note-taking"
    if any(a.startswith("electronics") for a in activity_set):
        return "electronics"
    return "idle"
