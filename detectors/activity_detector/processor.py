import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import PEN_LIKE, NOTE_SURFACE_LIKE
from app.utils.draw_utils import draw_labeled_box, draw_pose_skeleton
from app.utils.recognition_utils import (
    FACE_CASCADE,
    choose_primary_activity,
    extract_face_embedding,
    extract_head_point,
    face_recognition,
    match_face_id,
    person_hand_near_box,
    person_looking_at_box,
)
from app.utils.source_utils import is_youtube_url, resolve_source
from app.utils.vision_utils import bbox_from_landmarks, box_iou, box_to_box_distance, center_xy, normalize_list, point_to_box_distance

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    import torch
except ImportError:
    torch = None


def _init_hands(args):
    if not args.hands:
        return None, None, None
    if mp is None:
        raise RuntimeError("MediaPipe is not installed. Install with: pip install mediapipe==0.10.14")
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "mediapipe.solutions not found. "
            "Fix with: pip uninstall mediapipe -y && pip install mediapipe==0.10.14"
        )
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=args.max_hands,
        model_complexity=1,
        min_detection_confidence=args.hands_conf,
        min_tracking_confidence=args.hands_track,
    )
    return hands, mp_hands, mp_draw


def run_pipeline(args):
    def _safe_token(text):
        out = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
        out = out.strip("_")
        return out or "activity"

    def _conf_norm(conf, min_conf):
        if conf <= min_conf:
            return 0.0
        return float(np.clip((conf - min_conf) / max(1e-6, 1.0 - min_conf), 0.0, 1.0))

    def _head_near_box(person, box_xyxy, ratio):
        head = person.get("head")
        if head is None:
            return False
        px1, py1, px2, py2 = person["box"]
        person_size = max(1.0, max(px2 - px1, py2 - py1))
        d = float(np.linalg.norm(center_xy(box_xyxy) - head))
        return d <= ratio * person_size

    def _contact_score(a_box, b_box, max_dist):
        d = box_to_box_distance(a_box, b_box)
        return float(np.clip(1.0 - (d / max(1e-6, max_dist)), 0.0, 1.0))

    prompt_aliases = {
        "paper": ["sheet of paper", "loose leaf paper", "document", "printed handout", "notepad"],
        "sheet of paper": ["paper", "document", "handout"],
        "notebook": ["exercise notebook", "spiral notebook", "open notebook"],
        "worksheet": ["worksheet paper", "assignment sheet"],
        "book": ["textbook", "open book"],
        "writing": ["handwritten notes", "notes on paper"],
        "pen": ["ballpoint pen", "ink pen"],
        "pencil": ["wooden pencil", "mechanical pencil"],
        "cell phone": ["mobile phone", "smartphone", "phone"],
        "tablet": ["tablet computer", "ipad"],
        "laptop": ["laptop computer"],
    }
    pen_keywords = set(PEN_LIKE) | {"stylus", "marker", "highlighter"}
    electronics_keywords = {"laptop", "tablet", "tab", "phone", "cell phone", "mobile phone", "smartphone", "computer"}
    note_surface_keywords = set(NOTE_SURFACE_LIKE) | {
        "paper",
        "sheet of paper",
        "document",
        "handout",
        "notepad",
        "notebook",
        "worksheet",
        "book",
        "textbook",
        "writing",
        "notes",
    }

    def _expand_targets(targets):
        expanded = []
        seen_local = set()
        for t in targets:
            t = str(t).strip().lower()
            if not t:
                continue
            if t not in seen_local:
                seen_local.add(t)
                expanded.append(t)
            for alias in prompt_aliases.get(t, []):
                alias = str(alias).strip().lower()
                if alias and alias not in seen_local:
                    seen_local.add(alias)
                    expanded.append(alias)
        return expanded

    def _name_has_keyword(name, keywords):
        nm = str(name).lower().strip()
        return any((kw == nm) or (kw in nm) for kw in keywords)

    def _is_pen_name(name):
        return _name_has_keyword(name, pen_keywords)

    def _is_electronics_name(name):
        return _name_has_keyword(name, electronics_keywords)

    def _is_note_surface_name(name):
        nm = str(name).lower().strip()
        if _is_pen_name(nm):
            return False
        if _is_electronics_name(nm):
            return False
        return _name_has_keyword(nm, note_surface_keywords)

    activity_csv_path = Path(args.activity_out)
    download_dir_path = Path(args.download_dir)
    if not download_dir_path.is_absolute():
        download_dir_path = activity_csv_path.parent / download_dir_path

    resolved_source, cleanup_source_path = resolve_source(
        args.source,
        download_dir=download_dir_path,
        keep_downloaded_source=args.keep_downloaded_source,
    )
    if is_youtube_url(args.source):
        print(f"Resolved YouTube source to local file: {resolved_source}")

    inference_device = args.device
    if inference_device is None:
        if torch is not None and torch.cuda.is_available():
            inference_device = 0
        else:
            inference_device = "cpu"
    print(f"Inference device: {inference_device}")

    model = YOLO(args.weights)
    pose_model = YOLO(args.pose_weights)

    electronics_targets = normalize_list(args.electronics)
    note_targets = normalize_list(args.notetaking)
    expanded_electronics_targets = _expand_targets(electronics_targets)
    expanded_note_targets = _expand_targets(note_targets)

    classes = expanded_electronics_targets + expanded_note_targets
    seen = set()
    classes = [c for c in classes if not (c in seen or seen.add(c))]
    if not classes:
        raise RuntimeError("No object classes configured for YOLO-World. Provide electronics and/or note-taking classes.")
    model.set_classes(classes)

    names_map = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    normalized_names = {idx: str(name).lower() for idx, name in names_map.items()}

    note_surface_keywords.update(expanded_note_targets)
    electronics_keywords.update(expanded_electronics_targets)

    electronics_ids = {idx for idx, name in normalized_names.items() if _is_electronics_name(name)}
    note_ids = {idx for idx, name in normalized_names.items() if _is_note_surface_name(name) or _is_pen_name(name)}

    hands, mp_hands, mp_draw = _init_hands(args)

    cap = cv2.VideoCapture(resolved_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {resolved_source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 30.0
    step = max(1, int(round(fps_in)))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps_out,
        (width, height),
    )

    frame_idx = 0
    written = 0
    next_person_id = 1
    face_db = []
    last_boxes = {}
    last_seen = {}
    last_hand_center = {}
    writing_streak = defaultdict(int)
    primary_activity_frames = defaultdict(lambda: defaultdict(int))
    detailed_activity_frames = defaultdict(lambda: defaultdict(int))
    person_seen_frames = defaultdict(int)
    proof_links = {}

    proof_dir_path = Path(args.proof_dir)
    if not proof_dir_path.is_absolute():
        proof_dir_path = activity_csv_path.parent / proof_dir_path
    if proof_dir_path.exists():
        shutil.rmtree(proof_dir_path)
    proof_dir_path.mkdir(parents=True, exist_ok=True)

    def save_proof_link(
        summary_type,
        pid,
        activity,
        source_frame,
        person_box,
        source_frame_idx,
        object_box=None,
        object_label=None,
    ):
        key = (summary_type, int(pid), str(activity))
        if key in proof_links:
            return
        proof_frame = source_frame.copy()
        draw_labeled_box(proof_frame, person_box, f"P{pid}", (0, 255, 0), 2)
        if object_box is not None:
            obj_label = object_label if object_label is not None else "object"
            draw_labeled_box(proof_frame, object_box, obj_label, (0, 255, 255), 2)
        frame_dir = proof_dir_path / f"frame_{int(source_frame_idx):06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{summary_type}_p{pid}_{_safe_token(activity)}.jpg"
        fpath = frame_dir / fname
        cv2.imwrite(str(fpath), proof_frame)
        proof_links[key] = fpath.resolve().as_uri()

    def register_activity(person, activity, score, object_box=None, object_label=None):
        prev_score = person["activity_scores"].get(activity, -1.0)
        if score <= prev_score:
            return
        person["activity_scores"][activity] = float(score)
        person["activities"].add(activity)
        if object_box is not None:
            person["activity_objects"][activity] = {
                "box": object_box,
                "label": object_label if object_label is not None else "object",
            }

    if face_recognition is None:
        if FACE_CASCADE is not None:
            print("Warning: face_recognition not installed. Using OpenCV face-feature fallback for IDs.")
        else:
            print("Warning: no face-recognition backend found. Falling back to IoU tracking for person IDs.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue
        raw_frame = frame.copy()

        result = model.predict(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=inference_device,
            verbose=False,
        )[0]

        pose_result = pose_model.predict(
            frame,
            conf=args.pose_conf,
            imgsz=args.pose_imgsz,
            iou=args.pose_iou,
            max_det=args.pose_max_det,
            device=inference_device,
            verbose=False,
        )[0]

        people, objects = [], []

        if pose_result.boxes is not None and len(pose_result.boxes) > 0:
            p_xyxy = pose_result.boxes.xyxy.detach().cpu().numpy()
            p_conf = pose_result.boxes.conf.detach().cpu().numpy()
            kpts_xy = None
            kpts_conf = None
            if pose_result.keypoints is not None:
                kpts_xy = pose_result.keypoints.xy.detach().cpu().numpy()
                if pose_result.keypoints.conf is not None:
                    kpts_conf = pose_result.keypoints.conf.detach().cpu().numpy()

            for i, (box, score) in enumerate(zip(p_xyxy, p_conf)):
                if kpts_xy is not None and i < len(kpts_xy):
                    person_kpts_xy = kpts_xy[i]
                    if kpts_conf is not None and i < len(kpts_conf):
                        person_kpts_conf = kpts_conf[i]
                    else:
                        person_kpts_conf = np.ones((person_kpts_xy.shape[0],), dtype=np.float32)
                else:
                    person_kpts_xy = np.zeros((17, 2), dtype=np.float32)
                    person_kpts_conf = np.zeros((17,), dtype=np.float32)

                people.append({
                    "box": box,
                    "conf": float(score),
                    "kpts_xy": person_kpts_xy,
                    "kpts_conf": person_kpts_conf,
                    "head": extract_head_point(person_kpts_xy, person_kpts_conf, conf_thr=args.head_kpt_conf),
                    "hand_points": [],
                    "activities": set(),
                    "activity_scores": {},
                    "activity_objects": {},
                    "id": None,
                    "face_embedding": None,
                })

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            cls = result.boxes.cls.detach().cpu().numpy().astype(int)
            confs = result.boxes.conf.detach().cpu().numpy()

            for box, cls_id, score in zip(xyxy, cls, confs):
                det = {
                    "box": box,
                    "cls_id": cls_id,
                    "conf": float(score),
                    "name": normalized_names.get(cls_id, str(names_map.get(cls_id, "object")).lower()),
                }
                if cls_id in electronics_ids or cls_id in note_ids:
                    objects.append(det)

        for p in people:
            for wrist_idx in [9, 10]:
                if wrist_idx < len(p["kpts_conf"]) and p["kpts_conf"][wrist_idx] >= 0.35:
                    p["hand_points"].append(p["kpts_xy"][wrist_idx])

        detected_hands = []
        if hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                for hand_lms in res.multi_hand_landmarks:
                    lm_xy = [(lm.x, lm.y) for lm in hand_lms.landmark]
                    hand_box = bbox_from_landmarks(lm_xy, width, height, pad=12)
                    detected_hands.append({"box": hand_box, "landmarks": hand_lms})

        for h in detected_hands:
            hc = center_xy(h["box"])
            best_idx = None
            best_dist = 1e9
            for i, p in enumerate(people):
                d = point_to_box_distance(hc, p["box"])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is not None:
                px1, py1, px2, py2 = people[best_idx]["box"]
                person_size = max(1.0, max(px2 - px1, py2 - py1))
                if best_dist <= max(40.0, 0.25 * person_size):
                    people[best_idx]["hand_points"].append(hc)

        used_ids = set()
        for p in people:
            if face_recognition is not None or FACE_CASCADE is not None:
                emb = extract_face_embedding(frame, p["box"], face_model=args.face_model)
                p["face_embedding"] = emb
                pid = match_face_id(
                    emb,
                    face_db,
                    args.face_match_thresh,
                    used_ids,
                    fallback_face_cos_thresh=args.fallback_face_cos_thresh,
                )
                if pid is not None:
                    p["id"] = pid
                    used_ids.add(pid)

        reusable_prev_ids = [
            pid
            for pid in last_boxes.keys()
            if pid not in used_ids and (frame_idx - last_seen.get(pid, -999999)) <= (5 * step)
        ]

        for p in people:
            if p["id"] is not None:
                continue
            best_pid = None
            best_score = -1e9
            matched_by_rule = False
            for pid in reusable_prev_ids:
                iou = box_iou(p["box"], last_boxes[pid])
                cdist = float(np.linalg.norm(center_xy(p["box"]) - center_xy(last_boxes[pid])))
                lx1, ly1, lx2, ly2 = last_boxes[pid]
                prev_size = max(1.0, max(lx2 - lx1, ly2 - ly1))
                close_by_center = cdist <= (0.70 * prev_size)
                if not ((iou >= args.track_iou_thresh) or close_by_center):
                    continue
                score = iou - 0.0008 * cdist
                if score > best_score:
                    best_score = score
                    best_pid = pid
                    matched_by_rule = True
            if best_pid is not None and matched_by_rule:
                p["id"] = best_pid
                used_ids.add(best_pid)
                reusable_prev_ids.remove(best_pid)

        for p in people:
            if p["id"] is None:
                p["id"] = next_person_id
                next_person_id += 1
                used_ids.add(p["id"])

        for p in people:
            if p["face_embedding"] is not None:
                face_db.append({"id": p["id"], "embedding": p["face_embedding"]})
        if len(face_db) > 600:
            face_db = face_db[-600:]

        for p in people:
            last_boxes[p["id"]] = p["box"]
            last_seen[p["id"]] = frame_idx

        stale_ids = [pid for pid, fidx in last_seen.items() if (frame_idx - fidx) > (30 * step)]
        for pid in stale_ids:
            last_seen.pop(pid, None)
            last_boxes.pop(pid, None)
            last_hand_center.pop(pid, None)
            writing_streak.pop(pid, None)

        pens = [o for o in objects if _is_pen_name(o["name"]) and o["conf"] >= args.pen_obj_conf]
        note_surfaces = [o for o in objects if _is_note_surface_name(o["name"]) and o["conf"] >= args.note_obj_conf]
        weak_screen_like_note_candidates = []
        for o in objects:
            nm = o["name"]
            conf = o["conf"]
            is_laptop_like = "laptop" in nm
            is_tablet_like = ("tablet" in nm) or (nm == "tab") or ("ipad" in nm)
            # Only treat screen-like boxes as note fallback if they are below electronics confidence.
            if is_laptop_like and args.note_obj_conf <= conf < args.laptop_obj_conf:
                weak_copy = dict(o)
                weak_copy["from_weak_screen"] = True
                weak_screen_like_note_candidates.append(weak_copy)
            elif is_tablet_like and args.note_obj_conf <= conf < args.tablet_obj_conf:
                weak_copy = dict(o)
                weak_copy["from_weak_screen"] = True
                weak_screen_like_note_candidates.append(weak_copy)
        laptops = [
            o for o in objects
            if ("laptop" in o["name"]) and (o["conf"] >= args.laptop_obj_conf)
        ]
        phones = [o for o in objects if "phone" in o["name"] and o["conf"] >= args.phone_obj_conf]
        tablets = [
            o for o in objects
            if (("tablet" in o["name"]) or (o["name"] == "tab") or ("ipad" in o["name"]))
            and (o["conf"] >= args.tablet_obj_conf)
        ]

        for lap in laptops:
            if not people:
                continue
            lap_c = center_xy(lap["box"])
            pen_contact_any = 0.0
            if pens:
                pen_contact_any = max(_contact_score(pen["box"], lap["box"], args.contact_dist_px) for pen in pens)
            # Avoid classifying note materials as electronics when laptop confidence is weak
            # but pen/contact evidence around the box is strong.
            if pen_contact_any >= 0.45 and lap["conf"] < (args.laptop_obj_conf + 0.12):
                continue
            best_person = None
            best_score = -1.0
            best_meta = None
            for p in people:
                hand_near = 1.0 if person_hand_near_box(p, lap["box"]) else 0.0
                looking = 1.0 if person_looking_at_box(p, lap["box"]) else 0.0
                pen_in_hand = 1.0 if any(person_hand_near_box(p, pen["box"]) for pen in pens) else 0.0
                writing_conflict = 1.0 if (pen_in_hand and hand_near) else 0.0
                px1, py1, px2, py2 = p["box"]
                p_size = max(1.0, max(px2 - px1, py2 - py1))
                dist_score = float(np.clip(1.0 - (np.linalg.norm(lap_c - center_xy(p["box"])) / (1.2 * p_size)), 0.0, 1.0))
                lap_y = lap_c[1]
                in_torso_band = 1.0 if (py1 + 0.30 * (py2 - py1)) <= lap_y <= (py2 + 0.20 * (py2 - py1)) else 0.0
                conf_score = _conf_norm(lap["conf"], args.laptop_obj_conf)
                usage_score = (
                    0.34 * hand_near
                    + 0.22 * looking
                    + 0.22 * in_torso_band
                    + 0.12 * dist_score
                    + 0.10 * conf_score
                    - 0.16 * pen_contact_any
                    - 0.26 * writing_conflict
                )
                if usage_score > best_score:
                    best_score = usage_score
                    best_person = p
                    best_meta = {"writing_conflict": writing_conflict}
            if best_person is not None and best_score >= args.laptop_use_conf:
                if (
                    best_meta is not None
                    and best_meta["writing_conflict"] > 0.0
                    and lap["conf"] < (args.laptop_obj_conf + 0.14)
                ):
                    continue
                register_activity(best_person, "electronics:laptop", best_score, lap["box"], "laptop")
                cv2.line(
                    frame,
                    tuple(lap_c.astype(int)),
                    tuple(center_xy(best_person["box"]).astype(int)),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        for p in people:
            held_pens = [pen for pen in pens if person_hand_near_box(p, pen["box"])]
            has_pen_in_hand = len(held_pens) > 0
            px1, py1, px2, py2 = p["box"]
            p_size = max(1.0, max(px2 - px1, py2 - py1))
            hand_motion_px = 0.0
            if p["hand_points"]:
                hand_center = np.mean(np.asarray(p["hand_points"], dtype=np.float32), axis=0)
                prev_hand_center = last_hand_center.get(p["id"])
                if prev_hand_center is not None:
                    hand_motion_px = float(np.linalg.norm(hand_center - prev_hand_center))
                last_hand_center[p["id"]] = hand_center
            else:
                last_hand_center[p["id"]] = None

            best_phone = None
            best_phone_score = -1.0
            for ph in phones:
                phone_in_hand = person_hand_near_box(p, ph["box"])
                looking_phone = person_looking_at_box(p, ph["box"])
                head_near_phone = _head_near_box(p, ph["box"], args.phone_head_near_ratio)
                conf_score = _conf_norm(ph["conf"], args.phone_obj_conf)
                phone_score = (
                    0.45 * (1.0 if phone_in_hand else 0.0)
                    + 0.25 * (1.0 if looking_phone else 0.0)
                    + 0.20 * (1.0 if head_near_phone else 0.0)
                    + 0.10 * conf_score
                )
                if phone_score > best_phone_score:
                    best_phone_score = phone_score
                    best_phone = ph

            if best_phone is not None and best_phone_score >= args.phone_use_conf:
                register_activity(p, "electronics:phone", best_phone_score, best_phone["box"], best_phone["name"])
                cv2.line(
                    frame,
                    tuple(center_xy(best_phone["box"]).astype(int)),
                    tuple(center_xy(p["box"]).astype(int)),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            for tab in tablets:
                hand_on_tab = person_hand_near_box(p, tab["box"])
                looking_tab = person_looking_at_box(p, tab["box"])
                pen_tab_contact = 0.0
                if held_pens:
                    pen_tab_contact = max(_contact_score(pen["box"], tab["box"], args.contact_dist_px) for pen in held_pens)
                if pens:
                    pen_tab_contact = max(
                        pen_tab_contact,
                        0.85 * max(_contact_score(pen["box"], tab["box"], args.contact_dist_px) for pen in pens),
                    )
                conf_score = _conf_norm(tab["conf"], args.tablet_obj_conf)
                note_score = (
                    0.28 * (1.0 if has_pen_in_hand else 0.0)
                    + 0.34 * pen_tab_contact
                    + 0.18 * (1.0 if hand_on_tab else 0.0)
                    + 0.10 * (1.0 if looking_tab else 0.0)
                    + 0.10 * conf_score
                )
                elec_score = (
                    0.42 * (1.0 if hand_on_tab else 0.0)
                    + 0.26 * (1.0 if looking_tab else 0.0)
                    + 0.22 * (1.0 - pen_tab_contact)
                    + 0.10 * conf_score
                )
                if note_score >= args.tablet_note_use_conf:
                    register_activity(p, "note-taking:tablet", note_score, tab["box"], tab["name"])
                elif elec_score >= args.tablet_elec_use_conf:
                    if pen_tab_contact >= 0.35 and tab["conf"] < (args.tablet_obj_conf + 0.12):
                        continue
                    register_activity(p, "electronics:tablet", elec_score, tab["box"], tab["name"])

            best_note_surface = None
            best_note_score = -1.0
            best_note_meta = None
            note_candidates = list(note_surfaces)
            for weak_o in weak_screen_like_note_candidates:
                weak_copy = dict(weak_o)
                weak_copy["paper_label"] = "paper-like"
                weak_copy["note_conf"] = weak_o["conf"] * 0.75
                note_candidates.append(weak_copy)

            for n in note_candidates:
                looking_note = person_looking_at_box(p, n["box"])
                hand_on_note = person_hand_near_box(p, n["box"])
                pen_note_contact = 0.0
                if held_pens:
                    pen_note_contact = max(_contact_score(pen["box"], n["box"], args.contact_dist_px) for pen in held_pens)
                elif pens:
                    pen_note_contact = 0.75 * max(_contact_score(pen["box"], n["box"], args.contact_dist_px) for pen in pens)
                is_weak_screen = bool(n.get("from_weak_screen", False))
                # Weak screen fallbacks should require writing evidence; otherwise keep electronics behavior.
                if is_weak_screen and (pen_note_contact < 0.42):
                    continue
                motion_ref = max(6.0, 0.03 * p_size)
                motion_score = float(np.clip(hand_motion_px / motion_ref, 0.0, 1.0))
                writing_signal = (
                    0.42 * pen_note_contact
                    + 0.24 * (1.0 if hand_on_note else 0.0)
                    + 0.22 * (1.0 if has_pen_in_hand else 0.0)
                    + 0.12 * motion_score
                )
                conf_score = _conf_norm(n.get("note_conf", n["conf"]), args.note_obj_conf)
                note_score = (
                    0.26 * (1.0 if looking_note else 0.0)
                    + 0.30 * (1.0 if hand_on_note else 0.0)
                    + 0.24 * pen_note_contact
                    + 0.10 * (1.0 if has_pen_in_hand else 0.0)
                    + 0.10 * conf_score
                )
                if is_weak_screen:
                    note_score -= 0.10
                if note_score > best_note_score:
                    best_note_score = note_score
                    best_note_surface = n
                    best_note_meta = {
                        "pen_note_contact": float(pen_note_contact),
                        "hand_on_note": bool(hand_on_note),
                        "has_pen_in_hand": bool(has_pen_in_hand),
                        "motion_score": float(motion_score),
                        "writing_signal": float(writing_signal),
                        "is_weak_screen": bool(is_weak_screen),
                    }

            paper_threshold = args.paper_note_use_conf if has_pen_in_hand else max(0.50, args.paper_note_use_conf - 0.03)
            if best_note_surface is not None and best_note_meta is not None:
                strong_writing_frame = (
                    best_note_meta["hand_on_note"]
                    and (best_note_meta["pen_note_contact"] >= 0.36)
                    and (best_note_meta["has_pen_in_hand"] or best_note_meta["motion_score"] >= 0.50)
                )
                if strong_writing_frame:
                    writing_streak[p["id"]] = min(8, writing_streak[p["id"]] + 1)
                else:
                    writing_streak[p["id"]] = max(0, writing_streak[p["id"]] - 1)
                writing_ready = (
                    writing_streak[p["id"]] >= 2
                    or (strong_writing_frame and best_note_meta["writing_signal"] >= 0.74)
                )
                if best_note_meta["is_weak_screen"]:
                    writing_ready = (
                        writing_ready
                        and best_note_meta["pen_note_contact"] >= 0.50
                        and best_note_meta["writing_signal"] >= 0.68
                    )
                if writing_ready and best_note_score >= paper_threshold:
                    paper_label = best_note_surface.get("paper_label", best_note_surface["name"])
                    register_activity(p, "note-taking:paper", best_note_score, best_note_surface["box"], paper_label)
            else:
                writing_streak[p["id"]] = max(0, writing_streak[p["id"]] - 1)

        for o in objects:
            nm = o["name"]
            if any(k in nm for k in ["laptop", "phone", "tablet"]):
                color = (0, 255, 255)
                prefix = "electronic"
            elif _is_pen_name(nm):
                color = (0, 165, 255)
                prefix = "pen/pencil"
            else:
                color = (255, 0, 255)
                prefix = "note"
            draw_labeled_box(frame, o["box"], f"{prefix}: {nm} {o['conf']:.2f}", color, 2)

        for p in people:
            draw_pose_skeleton(frame, p["kpts_xy"], p["kpts_conf"], conf_thr=0.35)
            primary = choose_primary_activity(p["activities"])
            color = (255, 180, 0)
            if primary == "electronics":
                color = (0, 255, 0)
            elif primary == "note-taking":
                color = (255, 0, 255)
            draw_labeled_box(frame, p["box"], f"P{p['id']} {primary} {p['conf']:.2f}", color, 2)
            if p["head"] is not None:
                cv2.circle(frame, tuple(p["head"].astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)
            for hp in p["hand_points"]:
                cv2.circle(frame, tuple(np.asarray(hp, dtype=np.float32).astype(int)), 4, (0, 140, 255), -1, cv2.LINE_AA)

            person_seen_frames[p["id"]] += 1
            primary_activity_frames[p["id"]][primary] += 1
            if not p["activities"] and primary == "idle":
                detailed_activity_frames[p["id"]]["idle"] += 1
            else:
                for a in p["activities"]:
                    detailed_activity_frames[p["id"]][a] += 1

            primary_obj = None
            if primary != "idle":
                best_act = None
                best_act_score = -1.0
                for act_name, act_score in p["activity_scores"].items():
                    if act_name.startswith(f"{primary}:") and act_score > best_act_score:
                        best_act_score = act_score
                        best_act = act_name
                if best_act is not None:
                    primary_obj = p["activity_objects"].get(best_act)
            if primary != "idle":
                save_proof_link(
                    "primary",
                    p["id"],
                    primary,
                    raw_frame,
                    p["box"],
                    frame_idx,
                    object_box=None if primary_obj is None else primary_obj["box"],
                    object_label=None if primary_obj is None else primary_obj["label"],
                )
            if p["activities"]:
                for detail_activity in sorted(p["activities"]):
                    obj_info = p["activity_objects"].get(detail_activity)
                    save_proof_link(
                        "detailed",
                        p["id"],
                        detail_activity,
                        raw_frame,
                        p["box"],
                        frame_idx,
                        object_box=None if obj_info is None else obj_info["box"],
                        object_label=None if obj_info is None else obj_info["label"],
                    )

        for h in detected_hands:
            draw_labeled_box(frame, h["box"], "hand", (0, 140, 255), 2)
            if mp_draw is not None and mp_hands is not None:
                mp_draw.draw_landmarks(frame, h["landmarks"], mp_hands.HAND_CONNECTIONS)

        out.write(frame)
        written += 1
        frame_idx += 1

    cap.release()
    out.release()
    if hands is not None:
        hands.close()

    fps_out_safe = max(1e-6, float(args.fps_out))
    with open(activity_csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["summary_type", "person_id", "activity", "frames", "seconds", "proof_keyframe"])
        for pid in sorted(person_seen_frames.keys()):
            for activity, frames_count in sorted(primary_activity_frames[pid].items(), key=lambda kv: (-kv[1], kv[0])):
                seconds = frames_count / fps_out_safe
                proof_link = proof_links.get(("primary", int(pid), str(activity)), "")
                wcsv.writerow(["primary", pid, activity, frames_count, f"{seconds:.2f}", proof_link])
            for activity, frames_count in sorted(detailed_activity_frames[pid].items(), key=lambda kv: (-kv[1], kv[0])):
                seconds = frames_count / fps_out_safe
                proof_link = proof_links.get(("detailed", int(pid), str(activity)), "")
                wcsv.writerow(["detailed", pid, activity, frames_count, f"{seconds:.2f}", proof_link])

    print(f"Done. Wrote {written} frames at {args.fps_out} FPS. Output: {args.out}")
    print(f"Activity summary saved to: {activity_csv_path}")
    print(f"Proof keyframes saved to: {proof_dir_path}")
    if cleanup_source_path is not None:
        cleanup_path = Path(cleanup_source_path)
        if cleanup_path.exists():
            cleanup_path.unlink()
            print(f"Deleted temporary downloaded source: {cleanup_path}")
