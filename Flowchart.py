"""                   ┌─────────────────────────────┐
                   │        Start Frame Loop      │
                   └───────────────┬─────────────┘
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │ Read frame from video    │
                      │ (cap.read())             │
                      └───────────────┬──────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │ detect_people_and_     │
                         │ keypoints(frame)       │
                         └──────────────┬─────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────┐
                    │ Build detection list (bbox,     │
                    │ ankles, hips, body_mid)         │
                    └──────────────────┬──────────────┘
                                       │
                                       ▼
                     ┌────────────────────────────────┐
                     │   IoU Matching to Existing     │
                     │         Tracks                 │
                     │  (reuse track_id if IoU > 0.4) │
                     └──────────────────┬─────────────┘
                                        │
                                        ▼
        ┌─────────────────────────────────────────────────────────┐
        │ For detections with no match → create NEW track_id      │
        │ (NEXT_TRACK_ID++)                                       │
        └──────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
           ┌────────────────────────────────────────────────────┐
           │ Update track state:                                │
           │   last_seen, last_bbox, keypoints, ankle_mid,      │
           │   body_mid                                         │
           └────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
               ┌──────────────────────────────────────────────┐
               │ Assign rope using ankle position             │
               │ (assign_rope_below_point)                    │
               └──────────────────────────┬───────────────────┘
                                          │
                                          ▼
            ┌────────────────────────────────────────────────────┐
            │ INITIALIZATION PHASE (check if person started      │
            │ **above** the rope)                                │
            │                                                    │
            │ Conditions for init_confirm += 1:                  │
            │   - both ankles above rope_y                       │
            │   - both ankles confident                          │
            │   - repeat for INITIAL_CONFIRM_FRAMES (e.g., 2)   │
            └─────────────────────────┬──────────────────────────┘
                                      │
                            Yes       │       No
                        initialized?  │
                                      ▼
                          ┌──────────────────────────────┐
                          │ If not initialized → skip     │
                          │ fall logic this frame         │
                          └──────────────────────────────┘
                                      │
                                      ▼
        ┌──────────────────────────────────────────────────────┐
        │   FALL LOGIC (for initialized_above == True only)    │
        └──────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ Compute vertical score (median hips+shoulders)   │
          │ Smooth using EMA → smoothed_y                    │
          │ Compute downward_speed = smoothed_y - prev_y     │
          └────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
         ┌────────────────────────────────────────────────────┐
         │ Compute torso angle                                │
         │ Horizontal (<50°) → stronger fall evidence          │
         └────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────────────┐
                  │ Compare smoothed_y with rope height     │
                  │                                          │
                  │ is_below = smoothed_y > rope_y + T       │
                  │ T = FALL_THRESHOLD_PX (e.g., 15 px)      │
                  └───────────────────┬──────────────────────┘
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ Update fall_counter:                             │
             │   If is_below: +=1                               │
             │   If fast_drop (>4 px/frame): +=1                │
             │   If horizontal torso: +=1                       │
             │   Else (above rope): reset counters              │
             └──────────────────────────┬───────────────────────┘
                                        │
                                        ▼
                ┌─────────────────────────────────────────────┐
                │ If fall_counter ≥ FALL_PERSISTENCE_FRAMES   │
                │        → FALL CONFIRMED                     │
                └──────────────────────────┬───────────────────┘
                                           │
                                           ▼
                        ┌──────────────────────────────────┐
                        │ Save snapshot                    │
                        │ log_fall to CSV                  │
                        │ Mark state["is_fallen"] = True   │
                        └──────────────────────────────────┘
                                           │
                                           ▼
                      ┌───────────────────────────────────┐
                      │ Draw indicator for the person     │
                      │ (red = fallen, green = normal)    │
                      └───────────────────────────────────┘
                                           │
                                           ▼
                ┌────────────────────────────────────────────┐
                │ Cleanup: remove tracks not seen for ~60    │
                │ frames (≈ 4.7 seconds at 12.8 FPS)          │
                └────────────────────────────────────────────┘
                                           │
                                           ▼
                               Next Frame → repeat
"""