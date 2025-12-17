import os
import csv
import cv2
import math
from datetime import datetime

from pose_detector import detect_people_and_keypoints
from rope_utils import interpolate_rope_y

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
FALL_THRESHOLD_PX = 10
FALL_PERSISTENCE_FRAMES = 3
IOU_MATCH_THRESHOLD = 0.2
SNAPSHOT_ROOT = "snapshots"

MAX_MISSED_FRAMES = 60

# -------------------------------------------------
# Tracker globals
# -------------------------------------------------
NEXT_TRACK_ID = 0


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def smooth(prev, curr, alpha=0.5):
    if prev is None:
        return curr
    return alpha * curr + (1 - alpha) * prev


def compute_vertical_score(det):
    """
    Uses multiple body landmarks and returns a robust vertical position.
    """
    ys = []

    if det.get("left_shoulder"):
        ys.append(det["left_shoulder"][1])
    if det.get("right_shoulder"):
        ys.append(det["right_shoulder"][1])

    lh = det["hips"]["left"]
    rh = det["hips"]["right"]
    if lh:
        ys.append(lh[1])
    if rh:
        ys.append(rh[1])

    ys.append(det["body_mid"][1])

    ys.sort()
    return ys[len(ys) // 2]


def compute_torso_angle(det):
    """
    Angle of torso relative to horizontal.
    """
    ls = det.get("left_shoulder")
    rs = det.get("right_shoulder")
    lh = det["hips"]["left"]
    rh = det["hips"]["right"]

    if None in (ls, rs, lh, rh):
        return 90.0

    shoulder_mid = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5)
    hip_mid = ((lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5)

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]

    return abs(math.degrees(math.atan2(dy, dx)))


def iou(boxA, boxB):
    """
    Intersection over Union for two bboxes [x1,y1,x2,y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter_area / float(areaA + areaB - inter_area)


def create_person_state():
    return {
        "track_id": None,
        "rope_id": None,
        "fall_counter": 0,
        "above_counter": 0,
        "is_fallen": False,
        "prev_y": None,
        "smooth_y": None,
        "last_bbox": None,
        "last_seen": 0
    }


def save_snapshot(frame, camera_id, track_id):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")

    folder = os.path.join(SNAPSHOT_ROOT, date_str, camera_id)
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, f"{time_str}_person{track_id}.jpg")
    cv2.imwrite(path, frame)
    return path


def log_fall(camera_id, track_id, rope_id, snapshot_path):
    csv_path = f"fall_events_{camera_id}.csv"
    write_header = not os.path.exists(csv_path)

    now = datetime.now()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date", "time", "camera", "person", "rope", "snapshot"])
        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            camera_id,
            track_id,
            rope_id,
            snapshot_path
        ])


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------
def process_frame_for_falls(frame, camera_id, rope_polylines, tracked_persons, frame_idx):
    global NEXT_TRACK_ID

    output = frame.copy()
    detections = detect_people_and_keypoints(frame)

    matches = []

    # -------------------------
    # IoU-based tracking
    # -------------------------
    for det in detections:
        bbox = det["bbox"]
        best_id = None
        best_iou = 0.0

        for tid, state in tracked_persons.items():
            if state["last_bbox"] is None:
                continue
            val = iou(bbox, state["last_bbox"])
            if val > best_iou:
                best_iou = val
                best_id = tid

        if best_iou >= IOU_MATCH_THRESHOLD:
            matches.append((best_id, det))
        else:
            tid = NEXT_TRACK_ID
            NEXT_TRACK_ID += 1
            tracked_persons[tid] = create_person_state()
            tracked_persons[tid]["track_id"] = tid
            matches.append((tid, det))

    # -------------------------
    # Update tracks
    # -------------------------
    for tid, det in matches:
        state = tracked_persons[tid]
        state["last_seen"] = frame_idx
        state["last_bbox"] = det["bbox"]

        bx, by = det["body_mid"]
        x1, y1, x2, y2 = map(int, det["bbox"])
        bbox_height = max(1, y2 - y1)  # avoid division by zero

        # -------------------------
        # Rope assignment (once)
        # -------------------------
        if state["rope_id"] is None:
            ax, ay = det["ankle_mid"]
            best_rope = None
            best_gap = float("inf")

            for rid, rope in enumerate(rope_polylines):
                rope_y = interpolate_rope_y(ax, rope)
                if ay < rope_y and (rope_y - ay) < best_gap:
                    best_gap = rope_y - ay
                    best_rope = rid

            state["rope_id"] = best_rope

        if state["rope_id"] is None:
            continue

        rope_y = interpolate_rope_y(bx, rope_polylines[state["rope_id"]])

        # -------------------------
        # Vertical motion (scale-aware)
        # -------------------------
        raw_y = compute_vertical_score(det)
        sm_y = smooth(state["smooth_y"], raw_y)
        state["smooth_y"] = sm_y

        speed = 0.0 if state["prev_y"] is None else (sm_y - state["prev_y"])
        norm_speed = 0.0 if state["prev_y"] is None else (speed / bbox_height)

        state["prev_y"] = sm_y

        torso_angle = compute_torso_angle(det)

        # -------------------------
        # Adaptive thresholds (distance-aware)
        # -------------------------
        adaptive_fall_margin = max(FALL_THRESHOLD_PX, 0.15 * bbox_height)

        below_rope = sm_y > rope_y + adaptive_fall_margin
        fast_drop = (speed > 4) or (norm_speed > 0.08)
        horizontal = torso_angle < 50

        # -------------------------
        # Fall evidence accumulation
        # -------------------------
        if below_rope:
            if fast_drop:
                state["fall_counter"] += 2
            elif horizontal:
                state["fall_counter"] += 1
            else:
                state["fall_counter"] += 1
            state["above_counter"] = 0
        else:
            state["above_counter"] += 1
            if state["above_counter"] >= 6:
                state["fall_counter"] = 0
                state["is_fallen"] = False

        # -------------------------
        # Trigger fall
        # -------------------------
        fall_triggered = False
        if not state["is_fallen"] and state["fall_counter"] >= FALL_PERSISTENCE_FRAMES:
            state["is_fallen"] = True
            fall_triggered = True

        # -------------------------
        # Drawing
        # -------------------------
        color = (0, 0, 255) if state["is_fallen"] else (0, 255, 0)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.circle(output, (int(bx), int(by)), 6, color, -1)

        if state["is_fallen"]:
            cv2.putText(
                output,
                "FALL DETECTED!",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

        # -------------------------
        # Snapshot (after drawing)
        # -------------------------
        if fall_triggered:
            snapshot = save_snapshot(output, camera_id, tid)
            log_fall(camera_id, tid, state["rope_id"], snapshot)

    # -------------------------
    # Cleanup stale tracks
    # -------------------------
    for tid in list(tracked_persons.keys()):
        if frame_idx - tracked_persons[tid]["last_seen"] > MAX_MISSED_FRAMES:
            del tracked_persons[tid]

    return output, tracked_persons

