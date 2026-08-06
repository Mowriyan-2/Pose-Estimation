"""
Pose Angle Tracker — New MediaPipe Tasks API version
-----------------------------------------------------
Rewritten from the legacy `mp.solutions.pose` API to the new
`mediapipe.tasks.python.vision.PoseLandmarker` API, since the legacy
`solutions` module is being phased out and breaks on newer installs.

SETUP (one-time):
1. Install the current mediapipe version (Tasks API requires a recent build,
   NOT the 0.10.9 pin used for the legacy API):
       pip install mediapipe opencv-python numpy

2. Download the pose landmarker model file into the same folder as this
   script, named exactly "pose_landmarker.task":
       https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

   (You can also use the "full" or "heavy" variants for higher accuracy at
   the cost of speed — same URL pattern, swap "lite" for "full"/"heavy".)

Run:
    python pose_angle_tracker_tasks.py
"""

import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Landmark index constants
# The Tasks API does not expose the old mp_pose.PoseLandmark enum, so we
# define the indices ourselves (same 33-point layout as before).
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# ---------------------------------------------------------------------------
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# Skeleton connections for manual drawing (subset of full POSE_CONNECTIONS,
# enough for a clean overlay without needing the legacy drawing_utils).
POSE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]

# (label, point_a, vertex_point, point_c)
JOINTS_TO_TRACK = [
    ("L Elbow", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    ("R Elbow", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    ("L Shoulder", LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
    ("R Shoulder", RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
    ("L Hip", LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    ("R Hip", RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    ("L Knee", LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    ("R Knee", RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
]

VISIBILITY_THRESHOLD = 0.5
MODEL_PATH = "pose_landmarker.task"


def calculate_angle(a, b, c):
    """Return the angle (in degrees) at point b, formed by points a-b-c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_xy(landmarks, index):
    lm = landmarks[index]
    return [lm.x, lm.y], lm.visibility


def draw_skeleton(image, landmarks, frame_w, frame_h):
    for idx_a, idx_b in POSE_CONNECTIONS:
        pt_a, vis_a = get_xy(landmarks, idx_a)
        pt_b, vis_b = get_xy(landmarks, idx_b)
        if min(vis_a, vis_b) < VISIBILITY_THRESHOLD:
            continue
        px_a = tuple(np.multiply(pt_a, [frame_w, frame_h]).astype(int))
        px_b = tuple(np.multiply(pt_b, [frame_w, frame_h]).astype(int))
        cv2.line(image, px_a, px_b, (245, 117, 66), 2)

    for i, lm in enumerate(landmarks):
        if lm.visibility < VISIBILITY_THRESHOLD:
            continue
        px = tuple(np.multiply([lm.x, lm.y], [frame_w, frame_h]).astype(int))
        cv2.circle(image, px, 4, (245, 66, 230), -1)


def draw_angles(image, landmarks, frame_w, frame_h):
    for label, a_idx, b_idx, c_idx in JOINTS_TO_TRACK:
        a_xy, a_vis = get_xy(landmarks, a_idx)
        b_xy, b_vis = get_xy(landmarks, b_idx)
        c_xy, c_vis = get_xy(landmarks, c_idx)

        if min(a_vis, b_vis, c_vis) < VISIBILITY_THRESHOLD:
            continue

        angle = calculate_angle(a_xy, b_xy, c_xy)
        vertex_px = tuple(np.multiply(b_xy, [frame_w, frame_h]).astype(int))

        cv2.putText(image, f"{angle:.1f}°", vertex_px,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        label_px = (vertex_px[0], max(vertex_px[1] - 15, 0))
        cv2.putText(image, label, label_px,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)


def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (index 0). Check camera permissions/index.")
        return

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to read frame from camera.")
            break

        frame_h, frame_w = frame.shape[:2]

        # Tasks API expects RGB wrapped in an mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # VIDEO mode requires a strictly increasing timestamp (ms)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            # detect_for_video can return multiple detected people;
            # we use the first detected pose here.
            landmarks = result.pose_landmarks[0]
            draw_skeleton(frame, landmarks, frame_w, frame_h)
            draw_angles(frame, landmarks, frame_w, frame_h)

        cv2.imshow("Pose Angle Tracker (Tasks API)", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()