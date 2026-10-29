import cv2
import math
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.21
CLOSED_FRAMES_FOR_SLEEP = 30
MAX_FACES = 10  # adjust to how many students you expect in frame

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(landmarks, eye_points, w, h):
    coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]
    p1, p2, p3, p4, p5, p6 = coords
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_faces=MAX_FACES
)
landmarker = FaceLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Cannot open camera")
    exit()

# One closed-frame counter per face slot
closed_frame_counts = [0] * MAX_FACES

frame_timestamp_ms = 0
fps = camera.get(cv2.CAP_PROP_FPS) or 30
frame_duration_ms = int(1000 / fps)

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to capture video")
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += frame_duration_ms
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.face_landmarks:
            for i, landmarks in enumerate(result.face_landmarks):
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                

                if avg_ear < EAR_THRESHOLD:
                    closed_frame_counts[i] += 1
                else:
                    closed_frame_counts[i] = 0

                status = "SLEEPING" if closed_frame_counts[i] >= CLOSED_FRAMES_FOR_SLEEP else "ATTENTIVE"
                color = (0, 0, 255) if status == "SLEEPING" else (0, 255, 0)

                # Get a bounding area from the landmarks to draw a box + label
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, f"Face {i}: {status} (EAR {avg_ear:.2f})",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "No faces detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("EdithVision - Multi-Face Attentiveness", frame)

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()