import cv2
import threading
import queue
import time
import json
import os
import numpy as np
from typing import Optional, Tuple
from notifier import send_intrusion_alert
from ultralytics import YOLO

SENTINEL = object()

def load_polygon_from_zone_json(zone_json_path: str) -> np.ndarray:
    """
    Loads {"polygon": [[x1,y1],...,[x4,y4]]} from zone.json
    and returns OpenCV polygon array of shape (4,1,2), dtype int32.
    """
    with open(zone_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polygon = data.get("polygon", None)
    if polygon is None or len(polygon) != 4:
        raise ValueError(f"zone.json must contain polygon with exactly 4 points. Got: {polygon}")

    pts = np.array(polygon, dtype=np.int32)  # shape (4, 2)
    return pts.reshape((-1, 1, 2)).astype(np.int32)  # shape (4, 1, 2)

def scale_polygon_to_fit(pts: np.ndarray, target_w: float, target_h: float) -> np.ndarray:
    """
    Scale a 4-point polygon to fit within target width/height.
    Assumes pts are in the same origin (top-left) coordinate space as the frame.
    Returns int32 polygon shape (4,1,2).
    """
    if pts.size == 0:
        return pts.astype(np.int32)

    pts_float = pts.astype(np.float32)
    max_x = float(np.max(pts_float[:, 0]))
    max_y = float(np.max(pts_float[:, 1]))

    scale = 1.0
    if max_x > 0 and max_y > 0:
        scale = min(target_w / max_x, target_h / max_y, 1.0)

    scaled = (pts_float * scale).astype(np.int32)
    return scaled.reshape((-1, 1, 2))

def get_box_xyxy(box) -> Tuple[int, int, int, int]:
    """
    Extract integer (x1, y1, x2, y2) from an Ultralytics box element.
    """
    xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
    x1, y1, x2, y2 = xyxy
    return int(x1), int(y1), int(x2), int(y2)

def filter_results_to_person(results, person_class_id: int = 0):
    """
    Filter Ultralytics tracking results to ONLY keep COCO class 0 ("person").
    Modifies results[0].boxes in-place.
    """
    r0 = results[0]
    if getattr(r0, "boxes", None) is None or len(r0.boxes) == 0:
        return results

    cls = r0.boxes.cls  # tensor-like
    keep_mask = (cls == person_class_id)

    if keep_mask.sum().item() == 0:
        r0.boxes = r0.boxes[:0]
        return results

    r0.boxes = r0.boxes[keep_mask]
    return results

def reader_thread(
    cap: cv2.VideoCapture,
    q: "queue.Queue[object]",
    stop_event: threading.Event,
    debug: bool = False,
) -> None:
    """
    Background thread:
    - Reads frames from cv2.VideoCapture continuously.
    - Pushes frames into a bounded queue to avoid lag buildup.
    - Drops older frames if the queue is full (latest-frame preference).
    - Signals shutdown via stop_event + SENTINEL.
    - Includes a fixed 0.033s sleep to enforce 30 FPS pacing.
    """
    frame_count = 0
    try:
        if debug:
            print("[Reader] thread started")

        while not stop_event.is_set():
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                if debug:
                    print("[Reader] cap.read() returned ret=False (EOS or read failure).")
                break

            frame_count += 1

            try:
                q.put_nowait(frame)
            except queue.Full:
                # Drop oldest to keep latest frame
                try:
                    _ = q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    pass

            # Enforce a 30 FPS ingestion pace
            elapsed = time.time() - t_start
            sleep_needed = max(0.0, 0.033 - elapsed)
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            else:
                # If decoding took longer than 33ms, yield briefly
                time.sleep(0.001)
    except Exception as e:
        print("[Reader] Exception:", repr(e))
        stop_event.set()
    finally:
        cap.release()
        stop_event.set()
        try:
            q.put_nowait(SENTINEL)
        except queue.Full:
            pass
        if debug:
            print(f"[Reader] exiting. total frames read: {frame_count}")

def main() -> None:
    # Configuration
    video_path = "sample.mp4"
    zone_json_path = "zone.json"
    MODEL_WEIGHTS = "yolo11s.pt"  # Day 5: medium variant
    ALERTS_DIR = "alerts"

    QUEUE_MAXSIZE = 2
    FIRST_FRAME_TIMEOUT_S = 5.0
    LOOP_GET_TIMEOUT_S = 0.5
    TRACK_CONF = 0.25
    TRACK_IOU = 0.45
    PERSON_CLASS_ID = 0  # COCO: person

    # Prepare alerts directory
    if not os.path.exists(ALERTS_DIR):
        os.makedirs(ALERTS_DIR, exist_ok=True)

    # Load video and verify
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Read current frame resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load and scale zone polygon to current frame resolution
    try:
        base_poly = load_polygon_from_zone_json(zone_json_path)  # (4,1,2)
        # Scale polygon to fit current video frame resolution
        scaled_pts = scale_polygon_to_fit(base_poly.reshape(-1, 2).astype(np.int32), frame_width, frame_height)
        polygon_array = scaled_pts.reshape((4, 1, 2)).astype(np.int32)  # (4,1,2)
    except Exception as e:
        print(f"[Init] Warning: zone polygon load/scale failed: {e}")
        # Fallback: empty polygon (no intrusion checks)
        polygon_array = np.zeros((4, 1, 2), dtype=np.int32)

    # Load model
    model = YOLO(MODEL_WEIGHTS)

    # Prepare reader thread
    q: "queue.Queue[object]" = queue.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = threading.Event()

    t = threading.Thread(target=reader_thread, args=(cap, q, stop_event, False), daemon=True)
    t.start()

    print("[Main] GarudAI Day7: Robust Intrusion Loop (30 FPS). Press 'q' to quit.")

    latest_frame: Optional[any] = None
    first_frame_received = False
    first_deadline = time.time() + FIRST_FRAME_TIMEOUT_S

    # Overlay / visualization
    fps_smooth = None
    fps_alpha = 0.2
    prev_t = time.time()

    BLUE = (255, 0, 0)   # BGR
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    # Edge-state for intrusion
    zone_was_violated = False

    try:
        while not stop_event.is_set():
            try:
                item = q.get(timeout=LOOP_GET_TIMEOUT_S)
            except queue.Empty:
                if time.time() > first_deadline and not first_frame_received:
                    raise TimeoutError(
                        f"No frame received within {FIRST_FRAME_TIMEOUT_S}s. Check video path/codec."
                    )
                continue

            if item is SENTINEL:
                break

            if not first_frame_received:
                first_frame_received = True

            # Drain queue to keep newest frame
            latest_frame = item
            while True:
                try:
                    item2 = q.get_nowait()
                except queue.Empty:
                    break
                if item2 is SENTINEL:
                    stop_event.set()
                    break
                latest_frame = item2

            if latest_frame is None:
                continue

            # Inference: track humans
            results = model.track(
                source=latest_frame,
                persist=True,
                verbose=False,
                conf=TRACK_CONF,
                iou=TRACK_IOU,
            )

            # Keep only COCO 'person' class
            results = filter_results_to_person(results, person_class_id=PERSON_CLASS_ID)

            # Base visualization (boxes/IDs)
            plotted = results[0].plot()

            intruding_any = False
            r0 = results[0]
            ids = None
            if getattr(r0.boxes, "id", None) is not None:
                ids = r0.boxes.id

            # If there are detections, test each bounding box against the polygon
            if r0.boxes is not None and len(r0.boxes) > 0:
                for idx in range(len(r0.boxes)):
                    box = r0.boxes[idx]
                    x1, y1, x2, y2 = get_box_xyxy(box)

                    # Anchor points: head (top-center), center (dead-center), feet (bottom-center)
                    head = (int((x1 + x2) / 2.0), int(y1))
                    center_pt = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))
                    feet = (int((x1 + x2) / 2.0), int(y2))

                    anchor_points = [head, center_pt, feet]

                    inside_any = False
                    for pt in anchor_points:
                        if cv2.pointPolygonTest(polygon_array, pt, False) >= 0:
                            inside_any = True
                            intruding_any = True
                            # Emphasize this person on intrusion
                            cv2.rectangle(plotted, (x1, y1), (x2, y2), RED, 2)
                            # Mark the triggering anchor
                            cv2.circle(plotted, (pt[0], pt[1]), 4, GREEN, -1)

                            if ids is not None:
                                try:
                                    pid_val = ids[idx]
                                    pid = int(pid_val.item()) if hasattr(pid_val, "item") else int(pid_val)
                                    cv2.putText(
                                        plotted,
                                        f"person {pid}",
                                        (x1, max(0, y1 - 12)),
                                        FONT,
                                        0.7,
                                        RED,
                                        2,
                                        cv2.LINE_AA,
                                    )
                                except Exception:
                                    pass
                            break  # one intrusion per bbox is enough
                    # end for anchor_points
            # end if detections present

            # Draw the fence polygon (blue when safe, red when intruding)
            fence_color = RED if intruding_any else BLUE
            cv2.polylines(plotted, [polygon_array.reshape(-1, 2)], isClosed=True, color=fence_color, thickness=3)

            # Edge-triggered alert: save snapshot on first breach frame
            if intruding_any:
                if not zone_was_violated:
                    zone_was_violated = True
                    now_t = time.time()
                    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(now_t))
                    snapshot_path = os.path.join(ALERTS_DIR, f"breach-{ts}.jpg")
                    # Save the exact frame where breach was detected
                    try:
                        cv2.imwrite(snapshot_path, latest_frame)
                        print(f"[SYSTEM TRIGGER] Intrusion detected. Snapshot saved to {snapshot_path}")

                        send_intrusion_alert(ts, image_path=snapshot_path)

                    except Exception as e:
                        print(f"[SYSTEM TRIGGER] Failed to save snapshot: {e}")
            else:
                # Reset for the next breach
                zone_was_violated = False

            # FPS overlay
            now_t = time.time()
            dt = now_t - prev_t
            prev_t = now_t
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            if fps_smooth is None:
                fps_smooth = inst_fps
            else:
                fps_smooth = (1 - fps_alpha) * fps_smooth + fps_alpha * inst_fps

            cv2.putText(
                plotted,
                f"AI FPS: {fps_smooth:.1f}",
                (10, 30),
                FONT,
                0.9,
                GREEN,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Day7 - GarudAI Intrusion (YOLO11s)", plotted)

            #30 fps
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()
                break

    finally:
        stop_event.set()
        try:
            t.join(timeout=1.0)
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[Main] shutdown complete.")

if __name__ == "__main__":
    main()  