import cv2
import threading
import queue
import time
import json
import numpy as np
from typing import Optional, Tuple

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

    poly = np.array(polygon, dtype=np.int32)  # (4,2)
    return poly.reshape((-1, 1, 2)).astype(np.int32)  # (4,1,2)


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
    """
    frame_count = 0
    try:
        if debug:
            print("[Reader] thread started")

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if debug:
                    print("[Reader] cap.read() returned ret=False (EOS or read failure).")
                break

            frame_count += 1

            try:
                q.put_nowait(frame)
            except queue.Full:
                try:
                    _ = q.get_nowait()  # drop oldest
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    pass
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


def filter_results_to_person(results, person_class_id: int = 0):
    """
    Filter Ultralytics tracking results to ONLY keep COCO class 0 ("person").
    Modifies results[0].boxes in-place.
    """
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return results

    cls = r0.boxes.cls  # tensor-like
    keep_mask = (cls == person_class_id)

    if keep_mask.sum().item() == 0:
        r0.boxes = r0.boxes[:0]
        return results

    r0.boxes = r0.boxes[keep_mask]
    return results


def get_box_xyxy(box) -> Tuple[int, int, int, int]:
    """
    Extract integer (x1, y1, x2, y2) from an Ultralytics box element.
    """
    xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
    x1, y1, x2, y2 = xyxy
    return int(x1), int(y1), int(x2), int(y2)


def draw_fence(plotted, polygon_array_4x1x2: np.ndarray, color, thickness: int = 3):
    """
    Draws fence polygon lines + translucent fill.
    polygon_array_4x1x2: shape (4,1,2)
    """
    pts = polygon_array_4x1x2.reshape((-1, 2))

    cv2.polylines(
        plotted,
        [pts],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    overlay = plotted.copy()
    cv2.fillPoly(overlay, [pts], color=color)
    alpha = 0.10  # translucency
    cv2.addWeighted(overlay, alpha, plotted, 1 - alpha, 0, plotted)


def main() -> None:
    video_path = "sample.mp4"
    zone_json_path = "zone.json"

    # Day 4 -> Day 5: parameters
    QUEUE_MAXSIZE = 2
    FIRST_FRAME_TIMEOUT_S = 5.0
    LOOP_GET_TIMEOUT_S = 0.5

    # Speed/accuracy tuning from requirements
    MODEL_WEIGHTS = "yolo11s.pt"  # Day 5: medium variant
    PERSON_CLASS_ID = 0  # COCO: person
    TRACK_CONF = 0.25
    TRACK_IOU = 0.45

    # Load polygon fence
    polygon_array = load_polygon_from_zone_json(zone_json_path)
    polygon_array = np.ascontiguousarray(polygon_array, dtype=np.int32)

    # Load model once
    model = YOLO(MODEL_WEIGHTS)

    # Video capture (reader thread owns reads)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    q: "queue.Queue[object]" = queue.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = threading.Event()

    t = threading.Thread(
        target=reader_thread,
        args=(cap, q, stop_event, False),
        daemon=True,
    )
    t.start()

    print("[Main] Day5 running: Virtual Fence Intrusion + Tuned Tracking. Press 'q' to quit.")

    latest_frame: Optional[any] = None
    first_frame_received = False
    first_deadline = time.time() + FIRST_FRAME_TIMEOUT_S

    # FPS overlay
    fps_smooth = None
    fps_alpha = 0.2
    prev_t = time.time()

    # Visual colors (BGR)
    BLUE = (255, 0, 0)   # bright blue
    RED = (0, 0, 255)    # bright red

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    # Alert logging throttling (2 seconds)
    last_alert_log_t = 0.0
    ALERT_LOG_EVERY_S = 2.0

    # Main loop
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

            # Tracking inference (tuned sensitivity)
            results = model.track(
                source=latest_frame,
                persist=True,
                verbose=False,
                conf=TRACK_CONF,
                iou=TRACK_IOU,
            )

            # Person-only filtering
            results = filter_results_to_person(results, person_class_id=PERSON_CLASS_ID)

            # Base visualization (includes boxes/IDs if available)
            plotted = results[0].plot()

            intruding_any = False
            r0 = results[0]
            ids = None
            if getattr(r0.boxes, "id", None) is not None:
                ids = r0.boxes.id

            # Check each detected person using bottom-center point
            if r0.boxes is not None and len(r0.boxes) > 0:
                for idx in range(len(r0.boxes)):
                    box = r0.boxes[idx]
                    x1, y1, x2, y2 = get_box_xyxy(box)

                    x_center = (x1 + x2) / 2.0
                    y_bottom = (y1 + y2) / 2.0

                    inside = cv2.pointPolygonTest(polygon_array, (x_center, y_bottom), False)
                    if inside >= 0:
                        intruding_any = True

                        # Emphasize this person's box + optional asterisk
                        cv2.rectangle(plotted, (x1, y1), (x2, y2), RED, 2)
                        cv2.putText(
                            plotted,
                            "*",
                            (x1, max(0, y1 - 8)),
                            FONT,
                            1.0,
                            RED,
                            3,
                            cv2.LINE_AA,
                        )

                        # If IDs exist, label them
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
                                # Don't crash if ID formatting fails
                                pass

            # Draw fence overlay (blue safe, red intrusion)
            fence_color = RED if intruding_any else BLUE
            draw_fence(plotted, polygon_array, fence_color, thickness=3)

            # Warning banner + throttled alert logger
            if intruding_any:
                cv2.putText(
                    plotted,
                    "WARNING: INTRUSION DETECTED",
                    (10, 80),
                    FONT,
                    1.2,
                    RED,
                    4,
                    cv2.LINE_AA,
                )

                now_t = time.time()
                if (now_t - last_alert_log_t) >= ALERT_LOG_EVERY_S:
                    last_alert_log_t = now_t
                    # Timestamp format: 2026-06-03 20:15:00
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_t))
                    print(f"[ALERT] {ts} - Intrusion Detected in Restricted Zone!")

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
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Day5 - Virtual Fence Intrusion (YOLO11m tuned)", plotted)

            key = cv2.waitKey(30) & 0xFF
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
