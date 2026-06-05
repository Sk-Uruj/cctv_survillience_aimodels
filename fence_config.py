# fence_config.py
import json
import os
from typing import List, Tuple, Optional

import cv2

VIDEO_PATH = "sample.mp4"
ZONE_JSON_PATH = "zone.json"


def load_first_frame(video_path: str):
    """
    Opens the video, reads the first valid frame, then closes the capture.
    Returns the frame (numpy array) or raises RuntimeError if not possible.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # First valid frame
            return frame
    finally:
        cap.release()

    raise RuntimeError("Could not extract any frame from the video.")


def main():
    frame = load_first_frame(VIDEO_PATH)

    points: List[Tuple[int, int]] = []

    # We keep a separate copy for drawing each loop iteration.
    base_img = frame.copy()

    window_name = "Fence Zone Coordinate Picker (4-point polygon)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def draw_overlay(img):
        """
        Draw selected points and polygon overlay on a provided image (in-place).
        """
        # Draw points
        for (x, y) in points:
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)  # red dot (BGR)

        # Draw polygon lines when we have at least 2 points
        if len(points) >= 2:
            # For closed polygon, connect in order and wrap to first once we have 4
            if len(points) == 4:
                poly = points + [points[0]]
                for i in range(4):
                    cv2.line(img, poly[i], poly[i + 1], (0, 255, 0), 2)  # green edges
            else:
                # Not closed yet: draw lines between consecutive clicked points
                for i in range(len(points) - 1):
                    cv2.line(img, points[i], points[i + 1], (0, 255, 0), 2)

        # Helpful text
        cv2.putText(
            img,
            f"Left-click: add points ({len(points)}/4).  Keys: s=save, r=reset, q=quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def mouse_callback(event, x, y, flags, userdata):
        """
        On left click: store point and draw red dot immediately.
        """
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            # If already have 4 points, ignore further clicks until reset
            if len(points) >= 4:
                return

            points.append((int(x), int(y)))

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        # Start from base image each frame to keep overlay clean
        img = base_img.copy()
        draw_overlay(img)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            # Quit without saving
            break

        if key == ord("r"):
            # Reset selected points
            points.clear()

        if key == ord("s"):
            # Save only if we have exactly 4 points
            if len(points) != 4:
                print(f"Not enough points to save. Selected {len(points)}/4 points.")
                continue

            polygon = [[x, y] for (x, y) in points]
            config = {"polygon": polygon}

            # Print to terminal
            print(json.dumps(config, indent=2))

            # Save to zone.json
            with open(ZONE_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            print(f"Saved zone config to: {os.path.abspath(ZONE_JSON_PATH)}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
