import cv2

from detectors.activity_detector.config import COCO_SKELETON


def draw_labeled_box(frame, box_xyxy, label, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_pose_skeleton(frame, keypoints_xy, keypoints_conf, conf_thr=0.35):
    for xy, score in zip(keypoints_xy, keypoints_conf):
        if score < conf_thr:
            continue
        x, y = int(xy[0]), int(xy[1])
        cv2.circle(frame, (x, y), 3, (0, 220, 255), -1, cv2.LINE_AA)

    for a, b in COCO_SKELETON:
        if keypoints_conf[a] < conf_thr or keypoints_conf[b] < conf_thr:
            continue
        xa, ya = int(keypoints_xy[a][0]), int(keypoints_xy[a][1])
        xb, yb = int(keypoints_xy[b][0]), int(keypoints_xy[b][1])
        cv2.line(frame, (xa, ya), (xb, yb), (0, 220, 255), 2, cv2.LINE_AA)
