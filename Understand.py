import os
import csv
import cv2
from datetime import datetime


from pose_detector import detect_people_and_keypoints
from rope_utils import assign_rope_below_point, interpolate_rope_y


FALL_THRESHOLD_PX = 10
FALL_PERSISTENCE_FRAMES = 2
SNAPSHOT_ROOT = "snapshots"


def create_person_state():
   return {
       "rope_id": None,
       "fall_counter": 0,
       "above_counter": 0,    # <-- ADD THIS
       "is_fallen": False,
       "last_seen": 0
   }



def save_snapshot(frame, camera_id, person_id):
   date_str = datetime.now().strftime("%Y-%m-%d")
   time_str = datetime.now().strftime("%H-%M-%S")


   date_folder = os.path.join(SNAPSHOT_ROOT, date_str)
   os.makedirs(date_folder, exist_ok=True)


   camera_folder = os.path.join(date_folder, camera_id)
   os.makedirs(camera_folder, exist_ok=True)


   filename = f"{time_str}_person{person_id}.jpg"
   path = os.path.join(camera_folder, filename)


   cv2.imwrite(path, frame)
   return path


def log_fall(camera_id, person_id, rope_id, snapshot_path):
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
           person_id,
           rope_id,
           snapshot_path
       ])


def process_frame_for_falls(frame, camera_id, rope_polylines, tracked_persons, frame_idx):


   people = detect_people_and_keypoints(frame)
   output = frame.copy()


   for person in people:
       x1, y1, x2, y2 = person["bbox"]
       pid = person["id"]
       ankle_mid = person["ankle_mid"]
       body_mid  = person["body_mid"]


       if pid not in tracked_persons:
           tracked_persons[pid] = create_person_state()

 
       state = tracked_persons[pid]
       state["last_seen"] = frame_idx

     # Assign a rope below and closest to the ankle
       ax, ay = ankle_mid
       assigned_rope = assign_rope_below_point(ax, ay, rope_polylines)


       # Only update rope_id if a *valid* rope is detected
       if assigned_rope is not None:
           state["rope_id"] = assigned_rope


       # If STILL None, skip only if rope never assigned before
       if state["rope_id"] is None:
           continue


       assigned_rope = state["rope_id"]     # use locked rope


       bx, by = body_mid
       box_color = (0, 0, 255) if state["is_fallen"] else (0, 255, 0)
        # Draw bounding box
       cv2.rectangle(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            box_color,
            2
        )

       rope_y = interpolate_rope_y(bx, rope_polylines[assigned_rope])


       # Person is below the rope
       if by > rope_y + FALL_THRESHOLD_PX:
           state["fall_counter"] += 1
           state["above_counter"] = 0      # reset bounce counter


       # Person is above the rope (possible bounce)
       else:
           state["above_counter"] += 1


           # Only reset fall if above rope for too long (e.g., 3 frames)
           if state["above_counter"] >= 3:
               state["fall_counter"] = 0
               state["is_fallen"] = False




       if not state["is_fallen"] and state["fall_counter"] >= FALL_PERSISTENCE_FRAMES:
           state["is_fallen"] = True


           snapshot_path = save_snapshot(output, camera_id, pid)
           log_fall(camera_id, pid, assigned_rope, snapshot_path)
           cv2.putText(
            output,
            "FALL DETECTED!",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )

       color = (0,0,255) if state["is_fallen"] else (0,255,0)
       # Drawing the circle marker
       cv2.circle(output, (int(bx), int(by)), 6, color, -1)




   for pid in list(tracked_persons):
       if frame_idx - tracked_persons[pid]["last_seen"] > 60:
           del tracked_persons[pid]


   return output, tracked_persons


