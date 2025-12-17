# process_camera.py
import os
import sys
import json
import csv
import argparse
from datetime import datetime
import time

import cv2
import numpy as np
from ultralytics import YOLO
# Import your project modules (assumed to be in the same folder or PYTHONPATH)
import rope_drawer
import rope_utils
import pose_detector
import fall_logic
import psutil
process = psutil.Process(os.getpid())

# Default config path (same folder)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


# -------------------------
# Helpers: config + reconfig
# -------------------------
def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def reconfigure_pose_detector(config):
    """
    Load model from config["model_path"] and enforce confidence threshold
    by patching pose_detector._get_kp_xy to use the configured value.
    """
    # Load model if provided
    model_path = config.get("model_path")
    if model_path:
        try:
            pose_detector.model = YOLO(model_path)
            print(f"[process_camera] Loaded YOLO pose model from: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_path}': {e}")

    # Force confidence threshold used by _get_kp_xy
    conf_th = float(config.get("confidence_threshold", 0.30))
    # preserve original for fallback
    _orig_get_kp = getattr(pose_detector, "_get_kp_xy", None)

    def _get_kp_xy_with_conf(kp_array, index, conf_threshold=conf_th):
        if _orig_get_kp:
            return _orig_get_kp(kp_array, index, conf_threshold=conf_threshold)
        # fallback: basic extraction (shouldn't happen if pose_detector has the function)
        x, y, c = kp_array[index]
        return (float(x), float(y)) if c >= conf_threshold else None

    pose_detector._get_kp_xy = _get_kp_xy_with_conf
    print(f"[process_camera] Pose confidence threshold set to {conf_th}")


def reconfigure_fall_logic(config):
    """
    Apply settings to fall_logic (persistence frames) and monkeypatch
    its save_snapshot and log_fall functions so they honor config paths.
    """
    # persistence frames
    persistence = int(config.get("fall_confirm_frames", getattr(fall_logic, "FALL_PERSISTENCE_FRAMES", 1)))
    fall_logic.FALL_PERSISTENCE_FRAMES = persistence
    print(f"[process_camera] FALL_PERSISTENCE_FRAMES = {persistence}")

    snapshot_root = config.get("snapshot_folder", "fall_snapshots")
    csv_prefix = config.get("csv_prefix", "falls_")

    # allow save_snapshot to include frame index; process loop will set this attribute
    fall_logic._CURRENT_FRAME_IDX = None  # will be set by loop

    def save_snapshot(frame, camera_id, person_id):
        """
        Writes snapshot to:
          <snapshot_root>/<YYYY-MM-DD>/Camera-<camera_id>/snapshot_<HH-MM-SS>_<frame_idx>_person<person_id>.jpg
        Returns absolute path.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")

        date_folder = os.path.join(snapshot_root, date_str)
        os.makedirs(date_folder, exist_ok=True)

        camera_folder = os.path.join(date_folder, f"Camera-{camera_id}")
        os.makedirs(camera_folder, exist_ok=True)

        frame_idx = getattr(fall_logic, "_CURRENT_FRAME_IDX", "idx_unknown")
        filename = f"snapshot_{time_str}_{frame_idx}_person{person_id}.jpg"
        path = os.path.join(camera_folder, filename)

        # Write image (BGR expected)
        cv2.imwrite(path, frame)
        return os.path.abspath(path)

    def log_fall(camera_id, person_id, rope_id, snapshot_path):
        """
        CSV: <csv_prefix><camera_id>.csv
        Columns: iso_datetime, date, time, camera, person, rope, snapshot
        """
        csv_path = f"{csv_prefix}{camera_id}.csv"
        write_header = not os.path.exists(csv_path)
        now = datetime.now()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["iso_datetime", "date", "time", "camera", "person", "rope", "snapshot"])
            writer.writerow([
                now.isoformat(),
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                camera_id,
                person_id,
                rope_id,
                snapshot_path
            ])

    fall_logic.save_snapshot = save_snapshot
    fall_logic.log_fall = log_fall

    print(f"[process_camera] Monkeypatched fall_logic.save_snapshot -> root='{snapshot_root}' and log_fall -> prefix='{csv_prefix}'")


# -------------------------
# Drawing helpers
# -------------------------
def draw_ropes(frame, rope_polylines):
    """
    Draw rope polylines on the frame (green lines and points)
    """
    if not rope_polylines:
        return frame
    out = frame.copy()
    for rope in rope_polylines:
        if rope is None or len(rope) < 2:
            continue
        pts = np.array(rope, np.int32)
        cv2.polylines(out, [pts], False, (0, 200, 0), 2, lineType=cv2.LINE_AA)
        for (x, y) in pts:
            cv2.circle(out, (int(x), int(y)), 3, (0, 200, 0), -1)
    return out


def annotate_fall_text(frame, tracked_persons):
    """
    If any person has is_fallen True, place a top-left alert with counts.
    """
    fallen = sum(1 for s in tracked_persons.values() if s.get("is_fallen"))
    text = f"FALLS: {fallen}"
    cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


# -------------------------
# Main processing loop
# -------------------------
def run_video_loop(video_path, config, display=True, max_frames=None, save_output_video=True):
    camera_id = rope_drawer.extract_camera_id(video_path)
    print(f"[process_camera] Running camera_id = {camera_id} on video: {video_path}")

    # Load ropes
    rope_polylines = rope_utils.load_ropes_from_config(config, camera_id)
    print(f"[process_camera] Loaded {len(rope_polylines)} ropes for camera '{camera_id}' from config.json")

    if rope_polylines:
        print(f"[process_camera] Loaded {len(rope_polylines)} rope(s) for camera '{camera_id}'")
    else:
        print(f"[process_camera] No ropes found for camera '{camera_id}'. Continuing without rope assignment.")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[process_camera] Video opened: FPS={fps:.2f}, size=({width}x{height}), total_frames={total_frames}")

    # Prepare output video writer
    out_writer = None
    if save_output_video:
        out_name = f"output_{camera_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_name, fourcc, fps, (width, height))
        if not out_writer.isOpened():
            print(f"[process_camera] WARNING: could not open VideoWriter for '{out_name}'. Output video will not be saved.")
            out_writer = None
        else:
            print(f"[process_camera] Writing annotated output video to: {out_name}")

    tracked_persons = {}
    frame_idx = 0
    summary = {"frames": 0, "falls": 0, "snapshots": 0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[process_camera] End of video or read error.")
                break

            if max_frames is not None and frame_idx >= max_frames:
                print(f"[process_camera] Reached max_frames={max_frames}. Stopping.")
                break

            # Expose frame index for monkeypatched save_snapshot
            fall_logic._CURRENT_FRAME_IDX = frame_idx

            # Process frame: fall_logic handles detection + snapshot + csv logging
            start = time.time()

            annotated_frame, tracked_persons = fall_logic.process_frame_for_falls(
                frame, camera_id, rope_polylines, tracked_persons, frame_idx
            )
            
            end = time.time()
            inference_ms = (end - start) * 1000
            #fps = 1000 / inference_ms if inference_ms > 0 else 0

            #print(f"Inference: {inference_ms:.2f} ms  |  FPS: {fps:.2f}")

            #cpu_usage = process.cpu_percent(interval=0)
            #print(f"CPU Used: {cpu_usage:.2f}%")


            # Overlay rope lines and summary text (do after fall_logic's annotations)
            annotated_frame = draw_ropes(annotated_frame, rope_polylines)
            annotated_frame = annotate_fall_text(annotated_frame, tracked_persons)

            # Heuristic: count newly marked fallen persons in this frame
            new_falls = sum(
                1 for s in tracked_persons.values()
                if s.get("last_seen") == frame_idx and s.get("is_fallen", False)
            )
            if new_falls:
                summary["falls"] += new_falls
                summary["snapshots"] += new_falls  # save_snapshot is called once per fall transition

            summary["frames"] += 1

            # Show
            if display:
                cv2.imshow(f"Camera-{camera_id}", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[process_camera] Quit requested by user.")
                    break

            # Save annotated output video frame
            if out_writer is not None:
                # Ensure output is same size and BGR
                if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                    annotated_frame = cv2.resize(annotated_frame, (width, height))
                out_writer.write(annotated_frame)

            frame_idx += 1

    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if display:
            cv2.destroyAllWindows()

    print("[process_camera] Done.")
    print(f"[process_camera] Processed frames: {summary['frames']}, falls detected (heuristic): {summary['falls']}, snapshots (heuristic): {summary['snapshots']}")


# -------------------------
# CLI entrypoint
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Process camera video for fall detection (Module 5).")
    ap.add_argument("video", nargs="?", help="Video file path. If omitted, uses config.json video_path.")
    ap.add_argument("--config", "-c", default=CONFIG_PATH, help="Path to config.json")
    ap.add_argument("--no-display", action="store_true", help="Disable display window")
    ap.add_argument("--max-frames", type=int, default=None, help="Process only N frames (debug)")
    ap.add_argument("--no-output-video", dest="output_video", action="store_false", help="Disable writing annotated output video")
    args = ap.parse_args()

    config = load_config(args.config)
    video_path = args.video or config.get("video_path")
    if not video_path:
        raise RuntimeError("No video specified (CLI) and no video_path in config.json")

    # apply configuration
    reconfigure_pose_detector(config)
    reconfigure_fall_logic(config)

    # run
    run_video_loop(
        video_path,
        config,
        display=not args.no_display,
        max_frames=args.max_frames,
        save_output_video=args.output_video
    )


if __name__ == "__main__":
    main()
