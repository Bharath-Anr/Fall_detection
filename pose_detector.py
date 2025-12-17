import numpy as np
from ultralytics import YOLO
import cv2

# --------------------------------------------------------------------
# Load YOLO Pose model once (best practice)
# --------------------------------------------------------------------
model = YOLO("yolov8s-pose.pt")

# --------------------------------------------------------------------
# COCO keypoint indices used by YOLOv8 pose format
# --------------------------------------------------------------------
KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}

# --------------------------------------------------------------------
# Helper: Extract keypoint safely with confidence check
# --------------------------------------------------------------------

def _get_kp_xy(kp_array, index, conf_threshold=0.30):
    """
    Extracts the (x, y) coordinates of a keypoint only if its confidence
    is above a threshold. Returns None if the keypoint is unreliable.

    kp_array: shape (17, 3) -> [x, y, confidence]
    """
    x, y, conf = kp_array[index]
    if conf >= conf_threshold:
        return (float(x), float(y))
    return None

# --------------------------------------------------------------------
# Helper: Calculate midpoint of two points
# --------------------------------------------------------------------
def _midpoint(p1, p2):
    """
    Returns midpoint between two points (x1, y1) and (x2, y2).
    If either point is None, returns None.
    """
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)


# --------------------------------------------------------------------
# Main pose detection function
# --------------------------------------------------------------------
def detect_people_and_keypoints(frame):
    """
    Runs YOLOv8 pose detection on the frame and returns a list of persons.
    
    Each person has:
        - id:            index of detection (not tracked across frames)
        - keypoints:     raw YOLO keypoint array
        - ankles:        {left, right}
        - hips:          {left, right}
        - ankle_mid:     midpoint between ankles
        - hip_mid:       midpoint between hips
        - body_mid:      midpoint between hip_mid and ankle_mid
        - bbox:          bounding box [x1, y1, x2, y2]
        - score:         detection confidence

    Returns:
        List[dict] for each detected person.
        Persons lacking valid hips OR valid ankles are skipped.
    """
    results = model(frame, verbose=False)
    
    #annotated = results[0].plot()

    #cv2.imshow("Boxes", annotated)
    #cv2.waitKey(1)
    people = []

    # YOLOv8 pose results are in results[0].keypoints
    for det_id, (kp, box) in enumerate(zip(results[0].keypoints, results[0].boxes)):
        kp_array = kp.data[0].cpu().numpy()  # shape: (17, 3)

        # ------------------------------------------------------------
        # Extract hips & ankles with confidence checks
        # ------------------------------------------------------------
        left_ankle  = _get_kp_xy(kp_array, KP["left_ankle"])
        right_ankle = _get_kp_xy(kp_array, KP["right_ankle"])
        left_hip    = _get_kp_xy(kp_array, KP["left_hip"])
        right_hip   = _get_kp_xy(kp_array, KP["right_hip"])

        # Skip persons missing critical keypoints
        if left_ankle is None and right_ankle is None:
            continue
        if left_hip is None and right_hip is None:
            continue

        # ------------------------------------------------------------
        # Compute midpoints
        # ------------------------------------------------------------
        ankle_mid = _midpoint(left_ankle, right_ankle)
        hip_mid   = _midpoint(left_hip, right_hip)

        # Main point used in fall detection:
        # midpoint between hips midpoint and ankles midpoint
        body_mid = _midpoint(ankle_mid, hip_mid)

        # Safety check
        if ankle_mid is None or hip_mid is None or body_mid is None:
            continue

        # ------------------------------------------------------------
        # Parse bounding box
        # ------------------------------------------------------------
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        score = float(box.conf.cpu().numpy()[0])
        
        # ------------------------------------------------------------
        # Construct person structure
        # ------------------------------------------------------------
        person_info = {
            "id": det_id,
            "keypoints": kp_array,
            "hips": {
                "left": left_hip,
                "right": right_hip
            },
            "ankles": {
                "left": left_ankle,
                "right": right_ankle
            },  
            "ankle_mid": ankle_mid,
            "hip_mid": hip_mid,
            "body_mid": body_mid,  # used for fall detection logic
            "bbox": [x1, y1, x2, y2],
            "score": score
        }

        people.append(person_info)

    return people
