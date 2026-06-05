# Plan Coverage Review

This document checks the implementation against the project plan in `IOT_fin (2).pdf`.

## 1. Web Command Layer

Status: implemented.

- Streamlit command screen exists in `src/iot_servo_tracker/web/app.py`.
- Command packets are limited to `TRACK`, `STOP`, `CENTER`, and `REDETECT`.
- Hardware control is not run inside the Streamlit rerun loop.
- Duplicate command ids are rejected by the edge runtime.

Remaining integration work:

- Add authentication only if the web UI is exposed outside the internal lab network.

## 2. Edge/Server Split

Status: implemented.

- Raspberry Pi edge runtime owns camera capture, servo control, sensor validation, state transitions, and status ack.
- RTX server runtime owns vision inference and returns timestamped tracking packets.
- ZMQ frame/result channels are separated from MQTT command/status channels.

Remaining integration work:

- Tune ZMQ endpoints and MQTT broker settings on the actual network.

## 3. WeDetect-Ref Initial Detection

Status: implemented as Hugging Face local artifacts.

- The old external-service adapter has been removed.
- `HuggingFaceWeDetectRefClient` downloads `fushh7/WeDetect-Ref-2B` and the WeDetect-Uni checkpoint from `fushh7/WeDetect`.
- The local adapter receives `frame_bytes`, `query`, `wedetect_ref_model_dir`, and `wedetect_uni_checkpoint`.
- WeDetect-Ref is used for initial lock and redetection.

Remaining integration work:

- Provide the actual adapter module or script that wraps the official `WeChatCV/WeDetect` inference code.
- Choose 2B or 4B based on available RTX memory.

## 4. YOLO Tracking Handoff

Status: implemented.

- WeDetect result is used as the first lock.
- YOLO + tracker handles follow-up frames after a lock is established.
- The server preserves `locked_bbox` and `locked_track_id`.

Additional safeguards now implemented:

- Missing target and suspicious target are counted separately.
- High confidence alone does not accept a YOLO candidate.
- Center jump, track id switch with low IoU, bbox area growth, frame occupancy, and aspect ratio change are checked before servo control receives the bbox.

## 5. Similar Object Error

Status: implemented.

- Server rejects high-confidence candidates whose center jumps too far from the WeDetect/YOLO lock.
- Server rejects track id changes when IoU with the previous lock is too small.
- Edge sensor validator still detects center jump with nearly unchanged ToF distance as `SIMILAR_TARGET`.

Remaining risk:

- If a similar object appears in almost the same position, with similar size and distance, bbox geometry alone may not separate it. That needs appearance embedding or an explicit WeDetect-Ref recheck score.

## 6. Large Background/Object BBox Absorption

Status: implemented.

- Server rejects candidates that occupy too much of the frame.
- Server rejects candidates whose bbox area grows too much from the previous lock.
- Server rejects candidates whose aspect ratio changes too sharply.
- Edge safety validation has `BBOX_ABSORPTION` as a separate category.
- Safe-hold recovery does not trust the same `track_id` unless bbox continuity passes.

## 7. Delay Compensation

Status: implemented at lightweight level.

- Detection history estimates a current bbox from recent valid results.
- Stale inference packets are rejected before servo control.
- Communication delay can trigger `SAFE_HOLD`.

Remaining integration work:

- Tune thresholds with measured RTT from the actual RTX/Raspberry Pi network.

## 8. Sensor Validation

Status: implemented.

- ToF consistency is used for similar-target detection.
- Ultrasonic stability is used to reject sudden camera bbox jumps before servo control.
- Ultrasonic drop is used for occlusion detection.
- Infrared obstacle input immediately enters the safe-hold path and shows the Blue LED override.
- Limit switch input forces an error/safe stop path.
- Missing vision results count toward safe hold.

Remaining integration work:

- Calibrate ToF central-region trust against the actual camera/ToF mounting angle.

## 9. Servo Control And Safety

Status: implemented.

- PD control maps pixel error into pan/tilt servo command.
- Deadband, max speed, max acceleration, PWM mapping, axis limits, soft-stop, center return, and simulated servo driver exist.
- `SAFE_HOLD` uses powered soft-stop rather than immediate shutdown.
- Optional RGB status LED maps runtime states to visible colors, with Red for
  missing/rejected tracking and Blue for infrared obstacle detection.

Remaining integration work:

- Tune Kp/Kd, acceleration limits, and PWM bounds on the real servo frame.

## 10. Active Scan, Redetection, Limited Rescan

Status: implemented.

- Initial scan requires consecutive detections before lock.
- Scan failure returns to center.
- Safe hold can request redetection.
- Safe hold can transition to limited rescan, then center on timeout.

## 11. Camera Calibration / Distortion Correction

Status: partially implemented.

- Camera FOV geometry is implemented.
- Full OpenCV checkerboard calibration and undistort pipeline is not yet implemented.

Recommended next task:

- Add a calibration artifact file and apply `cv2.undistortPoints` before PD control.

## 12. Full Validation Entry Point

Status: implemented.

- `scripts/run_full_validation.py` runs unit tests plus focused smoke checks for external-service adapter removal, WeDetect-Ref configuration, similar-object rejection, large-bbox rejection, and edge safe-hold behavior.
