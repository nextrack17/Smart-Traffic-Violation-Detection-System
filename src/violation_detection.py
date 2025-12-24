import cv2
from ultralytics import YOLO

crossed_vehicles = set()
# ---------------- CONFIG ----------------
MODEL_PATH = "C:/Users/divya/cv_env/Traffic_Violation_Alert_System/models/traffic_violation_model4/weights/best.pt"
VIDEO_PATH = "C:/Users/divya/cv_env/Traffic_Violation_Alert_System/data/test_videos/red_light_violation.mp4"
OUTPUT_PATH = "outputs/violation_output.mp4"

VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle", "ambulance"]
RED_LIGHT_CLASS = "red_light"
GREEN_LIGHT_CLASS = "green_light"

CONF_THRES = 0.4
STOP_LINE_Y = 450          # adjust per video
RED_HOLD_FRAMES = 5        # red light persistence
COOLDOWN_FRAMES = 25       # prevent re-counting
# ----------------------------------------

print("🚦 Traffic Violation Detection Started")

# Load model
model = YOLO(MODEL_PATH)
print("✅ Model loaded")

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ ERROR: Could not open video")
    exit()
print("✅ Video opened")

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ---------------- STATE ----------------
violation_count = 0

recent_violations = []          # (cx, cy, frame_id)
previous_positions = {}         # vehicle_key -> prev_cy

red_light_frames = 0
red_light_active = False

frame_id = 0
# --------------------------------------

def recently_counted(cx, cy, current_frame, threshold=50):
    for vx, vy, f in recent_violations:
        if abs(cx - vx) < threshold and abs(cy - vy) < threshold \
           and current_frame - f < COOLDOWN_FRAMES:
            return True
    return False

# --------------- MAIN LOOP --------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # Clean old violation memory
    recent_violations = [
        (x, y, f) for (x, y, f) in recent_violations
        if frame_id - f < COOLDOWN_FRAMES
    ]

    results = model(frame, conf=CONF_THRES, verbose=False)[0]

    red_detected = False
    green_detected = False
    vehicles = []

    # -------- PARSE DETECTIONS --------
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_name == RED_LIGHT_CLASS:
            red_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        elif cls_name == GREEN_LIGHT_CLASS:
            green_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        elif cls_name in VEHICLE_CLASSES:
            vehicles.append((cls_name, x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # -------- SIGNAL STATE LOGIC --------
    if red_detected:
        red_light_frames += 1
    else:
        red_light_frames = max(0, red_light_frames - 1)

    red_light_active = red_light_frames >= RED_HOLD_FRAMES

    # Reset when green appears after red
    if green_detected and red_light_active:
        recent_violations.clear()
        previous_positions.clear()
        crossed_vehicles.clear()
        red_light_frames = 0
        red_light_active = False

    # -------- VIOLATION CHECK (CROSSING-BASED) ----------
    if red_light_active:
        for cls, x1, y1, x2, y2 in vehicles:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # quantized crossing ID (stable enough)
            crossing_id = (cls, cx // 35)
        
            # CROSSING CONDITION
            if STOP_LINE_Y - 15 < cy < STOP_LINE_Y + 15:
               if crossing_id not in crossed_vehicles:
                   violation_count += 1
                   crossed_vehicles.add(crossing_id)

                   cv2.putText(
                       frame,
                       "VIOLATION",
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 0, 255),
                       2,
                   )

    # -------- DRAW STOP LINE ----------
    cv2.line(
        frame,
        (0, STOP_LINE_Y),
        (width, STOP_LINE_Y),
        (0, 0, 255),
        2,
    )

    # -------- HUD ---------------------
    cv2.putText(
        frame,
        f"Violations: {violation_count}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
    )

    if red_light_active:
        cv2.putText(
            frame,
            "RED SIGNAL",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

    out.write(frame)
    cv2.imshow("Traffic Violation Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ------------- CLEANUP -----------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"🚨 Total Violations Detected: {violation_count}")


