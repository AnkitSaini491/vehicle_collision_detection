
import cv2
import time
import math
from collections import defaultdict, deque
from ultralytics import YOLO

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------

VIDEO_PATH = "input.mp4"
MODEL_PATH = "yolov8n.pt"

# COCO vehicle classes
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Approximate real-world widths in meters
REAL_WIDTH = {
    "Car": 1.8,
    "Motorcycle": 0.8,
    "Bus": 2.5,
    "Truck": 2.5
}

# Camera focal length
# Change this value after calibration for better accuracy
FOCAL_LENGTH = 700

# Warning distances
DANGER_DISTANCE = 8
CAUTION_DISTANCE = 15

# TTC thresholds
DANGER_TTC = 2
CAUTION_TTC = 4

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# -------------------------------------------------
# DATA FOR TRACKING
# -------------------------------------------------

previous_distance = {}
previous_time = {}

distance_history = defaultdict(lambda: deque(maxlen=10))

# FPS calculation
prev_frame_time = time.time()

# -------------------------------------------------
# DISTANCE FUNCTION
# -------------------------------------------------

def estimate_distance(pixel_width, object_type):

    if pixel_width <= 0:
        return 0

    real_width = REAL_WIDTH.get(object_type, 1.8)

    distance = (real_width * FOCAL_LENGTH) / pixel_width

    return distance


# -------------------------------------------------
# TTC FUNCTION
# -------------------------------------------------

def calculate_ttc(track_id, distance):

    current_time = time.time()

    if track_id not in previous_distance:
        previous_distance[track_id] = distance
        previous_time[track_id] = current_time

        return float("inf"), 0

    old_distance = previous_distance[track_id]
    old_time = previous_time[track_id]

    dt = current_time - old_time

    if dt <= 0:
        return float("inf"), 0

    # Positive means vehicle is getting closer
    closing_speed = (old_distance - distance) / dt

    previous_distance[track_id] = distance
    previous_time[track_id] = current_time

    if closing_speed <= 0:
        return float("inf"), closing_speed

    ttc = distance / closing_speed

    return ttc, closing_speed


# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    start_time = time.time()

    height, width = frame.shape[:2]

    # -------------------------------------------------
    # DETECTION + TRACKING
    # -------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=list(VEHICLE_CLASSES.keys()),
        conf=0.35,
        verbose=False
    )

    vehicle_count = 0

    nearest_distance = float("inf")
    nearest_ttc = float("inf")

    # -------------------------------------------------
    # DRAW ROAD / ROI
    # -------------------------------------------------

    # Approximate driving lane
    roi_left = int(width * 0.25)
    roi_right = int(width * 0.75)

    cv2.line(
        frame,
        (roi_left, height),
        (roi_left, int(height * 0.45)),
        (255, 255, 0),
        2
    )

    cv2.line(
        frame,
        (roi_right, height),
        (roi_right, int(height * 0.45)),
        (255, 255, 0),
        2
    )

    # -------------------------------------------------
    # PROCESS DETECTIONS
    # -------------------------------------------------

    if results[0].boxes is not None:

        boxes = results[0].boxes

        for i in range(len(boxes)):

            box = boxes[i]

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Class
            class_id = int(box.cls[0])

            if class_id not in VEHICLE_CLASSES:
                continue

            object_type = VEHICLE_CLASSES[class_id]

            vehicle_count += 1

            # Track ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = i

            # -------------------------------------------------
            # BOUNDING BOX WIDTH
            # -------------------------------------------------

            pixel_width = x2 - x1

            # Estimate distance
            distance = estimate_distance(
                pixel_width,
                object_type
            )

            # -------------------------------------------------
            # CENTER POINT
            # -------------------------------------------------

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # -------------------------------------------------
            # CHECK IF VEHICLE IS IN OUR LANE
            # -------------------------------------------------

            in_lane = (
                roi_left < center_x < roi_right
            )

            # -------------------------------------------------
            # TTC
            # -------------------------------------------------

            ttc, closing_speed = calculate_ttc(
                track_id,
                distance
            )

            # Save history
            distance_history[track_id].append(distance)

            # -------------------------------------------------
            # DECISION
            # -------------------------------------------------

            if in_lane:

                if distance < DANGER_DISTANCE or ttc < DANGER_TTC:

                    status = "DANGER"
                    box_color = (0, 0, 255)

                elif distance < CAUTION_DISTANCE or ttc < CAUTION_TTC:

                    status = "CAUTION"
                    box_color = (0, 165, 255)

                else:

                    status = "CLEAR"
                    box_color = (0, 255, 0)

            else:

                status = "OUT OF LANE"
                box_color = (255, 255, 0)

            # -------------------------------------------------
            # NEAREST VEHICLE
            # -------------------------------------------------

            if in_lane and distance < nearest_distance:

                nearest_distance = distance
                nearest_ttc = ttc

            # -------------------------------------------------
            # DRAW BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            # -------------------------------------------------
            # LABEL
            # -------------------------------------------------

            label = f"{object_type} ID:{track_id}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )

            # Distance
            distance_text = f"Distance: {distance:.1f} m"

            cv2.putText(
                frame,
                distance_text,
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2
            )

            # Status
            cv2.putText(
                frame,
                status,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            # TTC
            if math.isinf(ttc):

                ttc_text = "TTC: --"

            else:

                ttc_text = f"TTC: {ttc:.1f}s"

            cv2.putText(
                frame,
                ttc_text,
                (x1, y2 + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2
            )

    # -------------------------------------------------
    # FPS
    # -------------------------------------------------

    current_time = time.time()

    fps = 1 / max(
        current_time - prev_frame_time,
        0.001
    )

    prev_frame_time = current_time

    inference_time = (
        time.time() - start_time
    ) * 1000

    # -------------------------------------------------
    # TOP DASHBOARD
    # -------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (350, 155),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "VEHICLE SAFETY SYSTEM",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Inference: {inference_time:.1f} ms",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # -------------------------------------------------
    # MAIN SAFETY DECISION
    # -------------------------------------------------

    if nearest_distance == float("inf"):

        overall_status = "CLEAR"
        status_color = (0, 255, 0)

    elif (
        nearest_distance < DANGER_DISTANCE
        or nearest_ttc < DANGER_TTC
    ):

        overall_status = "DANGER"
        status_color = (0, 0, 255)

    elif (
        nearest_distance < CAUTION_DISTANCE
        or nearest_ttc < CAUTION_TTC
    ):

        overall_status = "CAUTION"
        status_color = (0, 165, 255)

    else:

        overall_status = "CLEAR"
        status_color = (0, 255, 0)

    # -------------------------------------------------
    # STATUS PANEL
    # -------------------------------------------------

    cv2.rectangle(
        frame,
        (width - 300, 10),
        (width - 10, 100),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "SAFETY STATUS",
        (width - 280, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        overall_status,
        (width - 280, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        3
    )

    # -------------------------------------------------
    # NEAREST VEHICLE INFORMATION
    # -------------------------------------------------

    if nearest_distance != float("inf"):

        cv2.putText(
            frame,
            f"Nearest: {nearest_distance:.1f} m",
            (width - 300, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        if math.isinf(nearest_ttc):

            ttc_display = "--"

        else:

            ttc_display = f"{nearest_ttc:.1f}s"

        cv2.putText(
            frame,
            f"TTC: {ttc_display}",
            (width - 300, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # -------------------------------------------------
    # SHOW FRAME
    # -------------------------------------------------

    cv2.imshow(
        "Vehicle Collision Detection System",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -------------------------------------------------
# CLEANUP
# -------------------------------------------------

cap.release()
cv2.destroyAllWindows()
