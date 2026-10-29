import csv
import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    ObjectDetector, ObjectDetectorOptions,
    PoseLandmarker, PoseLandmarkerOptions, RunningMode
)
from tracker import CentroidTracker

WARMUP_SECONDS = 5       # time to count the people already in class
LEAVE_FRAMES = 90        # about 3 seconds at 30 fps before someone counts as "left"
PERSON_MIN_SCORE = 0.5   # people need higher confidence than phones
PHONE_MIN_SCORE = 0.3
HEAD_DOWN_RATIO = 0.25   # below this ratio = head is down
HEAD_DOWN_SECONDS = 5    # how long before it counts as resting
PHONE_SECONDS = 3        # phone must stay with a person this long to count
PHONE_GRACE = 1.0        # phone detection flickers, so allow 1 second of misses
REVIEW_FRACTION = 0.2    # status becomes REVIEW if head-down + phone time is above this share

LOG_FILE = "attendance_log.csv"
SUMMARY_FILE = "attendance_summary.csv"


def iou_and_overlap(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0
    overlap = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0
    return iou, overlap


def remove_duplicates(dets):
    """dets: list of (score, box). Keeps the highest-scoring box when
    two boxes cover the same person."""
    dets = sorted(dets, key=lambda d: d[0], reverse=True)
    kept = []
    for score, box in dets:
        duplicate = False
        for _, kept_box in kept:
            iou, overlap = iou_and_overlap(box, kept_box)
            if iou > 0.5 or overlap > 0.85:
                duplicate = True
                break
        if not duplicate:
            kept.append((score, box))
    return [box for _, box in kept]


def shoulder_center(pose, fw, fh):
    return ((pose[11].x + pose[12].x) / 2 * fw,
            (pose[11].y + pose[12].y) / 2 * fh)


def head_ratio(pose, fw, fh):
    """How far the nose is above the shoulder line, in shoulder widths."""
    shoulder_w = abs(pose[11].x - pose[12].x) * fw
    if shoulder_w < 1:
        return None
    mid_y = (pose[11].y + pose[12].y) / 2 * fh
    return (mid_y - pose[0].y * fh) / shoulder_w


# ---------- Logging ----------
log_handle = open(LOG_FILE, "w", newline="", encoding="utf-8")
log_writer = csv.writer(log_handle)
log_writer.writerow(["timestamp", "id", "event"])

stats = {}  # id -> time totals for the summary file


def log_event(event_log, obj_id, kind):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_writer.writerow([stamp, obj_id, kind])
    log_handle.flush()
    line = f"{time.strftime('%H:%M:%S')}  ID {obj_id} {kind}"
    print(line)
    event_log.append(line)


def get_stats(obj_id, now):
    if obj_id not in stats:
        stats[obj_id] = {"first_seen": now, "last_seen": now,
                         "visible": 0.0, "head_down": 0.0, "phone": 0.0}
    return stats[obj_id]


def write_summary():
    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "first_seen", "last_seen", "visible_seconds",
                         "head_down_seconds", "phone_seconds",
                         "attentive_seconds", "attentive_percent", "status"])
        for obj_id, s in sorted(stats.items()):
            visible = s["visible"]
            flagged = s["head_down"] + s["phone"]
            attentive = max(visible - flagged, 0)
            percent = (attentive / visible * 100) if visible > 0 else 0
            status = "REVIEW" if visible > 0 and flagged / visible > REVIEW_FRACTION else "OK"
            writer.writerow([
                obj_id,
                time.strftime("%H:%M:%S", time.localtime(s["first_seen"])),
                time.strftime("%H:%M:%S", time.localtime(s["last_seen"])),
                round(visible, 1), round(s["head_down"], 1), round(s["phone"], 1),
                round(attentive, 1), round(percent, 1), status
            ])


# ---------- Models ----------
object_options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path="efficientdet_lite0.tflite"),
    running_mode=RunningMode.VIDEO,
    score_threshold=0.3,
    category_allowlist=["person", "cell phone"],
    max_results=50
)
detector = ObjectDetector.create_from_options(object_options)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=RunningMode.VIDEO,
    num_poses=10
)
pose_detector = PoseLandmarker.create_from_options(pose_options)

tracker = CentroidTracker(max_disappeared=LEAVE_FRAMES, max_distance=150)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Cannot open camera")
    exit()

frame_timestamp_ms = 0
fps = camera.get(cv2.CAP_PROP_FPS) or 30
frame_duration_ms = int(1000 / fps)

start_time = time.time()
last_frame_time = start_time
baseline = None
event_log = []

head_down_since = {}       # id -> time the head went down
head_down_alerted = set()  # ids already alerted
phone_since = {}           # id -> time the phone first appeared with this person
phone_last_seen = {}       # id -> last time a phone was seen with this person
phone_alerted = set()      # ids already alerted

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to capture video")
            break

        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += frame_duration_ms
        result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        pose_result = pose_detector.detect_for_video(mp_image, frame_timestamp_ms)

        person_dets = []
        phone_boxes = []

        for detection in result.detections:
            label = detection.categories[0].category_name
            score = detection.categories[0].score
            box = detection.bounding_box
            b = (box.origin_x, box.origin_y, box.width, box.height)
            if label == "person" and score >= PERSON_MIN_SCORE:
                person_dets.append((score, b))
            elif label == "cell phone" and score >= PHONE_MIN_SCORE:
                phone_boxes.append(b)

        person_boxes = remove_duplicates(person_dets)
        tracked = tracker.update(person_boxes)

        visible_ids = [i for i in tracked if tracker.disappeared[i] == 0]
        now_count = len(visible_ids)
        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        elapsed = now - start_time

        # Decide which person each phone belongs to
        ids_with_phone = set()
        for (px, py, pw, ph) in phone_boxes:
            pcx, pcy = px + pw / 2, py + ph / 2
            best_id, best_dist = None, None
            for obj_id in visible_ids:
                x, y, w, h = tracked[obj_id]
                if x - 20 <= pcx <= x + w + 20 and y - 20 <= pcy <= y + h + 20:
                    d = math.hypot(pcx - (x + w / 2), pcy - (y + h / 2))
                    if best_dist is None or d < best_dist:
                        best_id, best_dist = obj_id, d
            if best_id is not None:
                ids_with_phone.add(best_id)

        # Forget timers for people who are no longer visible
        for old_id in list(head_down_since.keys()):
            if old_id not in visible_ids:
                head_down_since.pop(old_id, None)
                head_down_alerted.discard(old_id)
        for old_id in list(phone_since.keys()):
            if old_id not in visible_ids:
                phone_since.pop(old_id, None)
                phone_last_seen.pop(old_id, None)
                phone_alerted.discard(old_id)

        # Check each person and draw them
        for obj_id in visible_ids:
            x, y, w, h = tracked[obj_id]

            # --- Head down check ---
            ratio = None
            for pose in pose_result.pose_landmarks:
                cx, cy = shoulder_center(pose, frame_w, frame_h)
                if x <= cx <= x + w and y <= cy <= y + h:
                    ratio = head_ratio(pose, frame_w, frame_h)
                    break

            if ratio is not None:
                if ratio < HEAD_DOWN_RATIO:
                    head_down_since.setdefault(obj_id, now)
                else:
                    head_down_since.pop(obj_id, None)
                    head_down_alerted.discard(obj_id)

            head_down = False
            down_secs = 0
            if obj_id in head_down_since:
                down_secs = now - head_down_since[obj_id]
                if down_secs >= HEAD_DOWN_SECONDS:
                    head_down = True
                    if obj_id not in head_down_alerted:
                        head_down_alerted.add(obj_id)
                        log_event(event_log, obj_id, "head down")

            # --- Phone check ---
            if obj_id in ids_with_phone:
                phone_last_seen[obj_id] = now
                phone_since.setdefault(obj_id, now)
            elif obj_id in phone_since and now - phone_last_seen[obj_id] > PHONE_GRACE:
                phone_since.pop(obj_id, None)
                phone_last_seen.pop(obj_id, None)
                phone_alerted.discard(obj_id)

            using_phone = obj_id in phone_since and now - phone_since[obj_id] >= PHONE_SECONDS
            if using_phone and obj_id not in phone_alerted:
                phone_alerted.add(obj_id)
                log_event(event_log, obj_id, "using phone")

            # --- Update totals for the summary ---
            s = get_stats(obj_id, now)
            s["last_seen"] = now
            s["visible"] += dt
            if head_down:
                s["head_down"] += dt
            if using_phone:
                s["phone"] += dt

            # --- Drawing ---
            statuses = []
            color = (0, 255, 0)
            if head_down:
                statuses.append(f"HEAD DOWN {int(down_secs)}s")
                color = (0, 140, 255)
            if using_phone:
                statuses.append("PHONE")
                color = (0, 0, 255)

            label = f"ID {obj_id}"
            if ratio is not None:
                label += f"  r={ratio:.2f}"
            if statuses:
                label += "  " + " + ".join(statuses)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw phone boxes
        for (x, y, w, h) in phone_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "phone", (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if elapsed < WARMUP_SECONDS:
            remaining = WARMUP_SECONDS - int(elapsed)
            cv2.putText(frame, f"Calibrating... {remaining}s (stay in frame)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            if baseline is None:
                baseline = now_count
                print(f"Baseline set: {baseline} people")

            for kind, obj_id in tracker.events:
                log_event(event_log, obj_id, kind)
            event_log = event_log[-5:]

            cv2.putText(frame, f"Now: {now_count}   Baseline: {baseline}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if now_count > baseline:
                cv2.putText(frame, f"EXTRA PEOPLE: +{now_count - baseline}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif now_count < baseline:
                cv2.putText(frame, f"MISSING: {baseline - now_count}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            for k, line in enumerate(event_log):
                cv2.putText(frame, line, (10, frame_h - 15 - 25 * (len(event_log) - 1 - k)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("EdithVision - Headcount Test", frame)

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
    try:
        write_summary()
        print(f"Saved {LOG_FILE} and {SUMMARY_FILE}")
    except PermissionError:
        print("Could not save the summary. Close attendance_summary.csv if it is open, then run again.")
    log_handle.close()