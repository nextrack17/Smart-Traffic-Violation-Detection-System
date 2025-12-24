## 📁 Project Structure

```text

Smart-Traffic-Violation-Detection-System/
├── src/
│   ├── train.py          # Trains YOLOv8 model on traffic dataset
│   └── test_video.py     # Runs inference on video and detects violations
├── data/                 # Dataset directory (not included in repo)
├── models/               # Trained weights (not included in repo)
├── outputs/              # Inference outputs (not included in repo)
├── requirements.txt
├── README.md
└── .gitignore

Summary: A real-time red-light violation detection pipeline using YOLOv8 and OpenCV. Combines deep learning with rule-based logic for robust violation analysis with low false positives.

## ⚙️ Environment & Requirements

- Python 3.10
- PyTorch 2.x
- Ultralytics YOLOv8
- OpenCV
- NumPy
- NVIDIA GPU RTX3050 with CUDA support

All dependencies are listed in `requirements.txt`.



## 📊 Results & Metrics

The model was trained using YOLOv8 on a merged traffic surveillance dataset containing vehicles and traffic signal classes. Performance was evaluated on a held-out validation set.

### 🔹 Object Detection Performance

| Metric        | Value |
|---------------|-------|
| Precision (P) | 0.738 |
| Recall (R)    | 0.690 |
| mAP@50        | 0.722 |
| mAP@50–95     | 0.518 |

### 🔹 Per-Class mAP@50 (Selected)

| Class         | mAP@50 |
|---------------|--------|
| Car           | 0.769  |
| Bus           | 0.608  |
| Truck         | 0.453  |
| Motorcycle    | 0.517  |
| Red Light     | 0.886  |
| Green Light   | 0.843  |
| Yellow Light  | 0.988  |

Traffic signal classes show strong performance, which is critical for reliable violation detection.

---

### 🚦 Traffic Violation Detection Results

The trained model was deployed on real-world traffic video footage. A rule-based logic layer was applied on top of detections to identify red-light violations based on:

- Traffic signal state (Red)
- Vehicle presence
- Stop-line crossing during red phase

**Detected Violations:** `8`

Violations are sparse events in real traffic scenarios. The system prioritizes correctness and low false positives over inflated counts, making it suitable for real-world deployment and analysis.

---

### ⚡ Inference Performance

- Average inference time: ~37 ms per frame (RTX 3050 Laptop GPU)
- Real-time capable for traffic surveillance applications


## 🧠 Methodology

The system follows a modular, end-to-end pipeline for detecting traffic violations from video streams. The methodology is designed to be interpretable, robust, and close to real-world deployment scenarios.

### 1. Object Detection
- A YOLOv8 object detection model is trained to detect:
  - Vehicles: `car`, `bus`, `truck`, `motorcycle`, `ambulance`
  - Traffic signals: `red_light`, `green_light`, `yellow_light`
- The model processes each video frame independently and outputs bounding boxes with class labels and confidence scores.

### 2. Traffic Signal State Identification
- Traffic light detections are analyzed per frame.
- The dominant signal state (Red / Yellow / Green) is determined based on:
  - Detection confidence
  - Spatial consistency of signal bounding boxes
- Only **red signal states** are considered for violation analysis.

### 3. Virtual Stop Line Definition
- A virtual stop line is manually defined in the scene using fixed pixel coordinates.
- This line represents the legal stopping boundary at an intersection.
- The stop line remains constant throughout the video.

### 4. Vehicle–Stop Line Interaction Analysis
- For each detected vehicle:
  - The bottom-center point of the bounding box is tracked per frame.
  - The system checks whether this point crosses the stop line.
- Crossing events are evaluated **only when the traffic signal is red**.

### 5. Violation Logic
A traffic violation is registered if and only if:
- The traffic signal is **red**, and
- A vehicle crosses the predefined stop line.

This rule-based logic significantly reduces false positives and ensures that only genuine violations are counted.

### 6. Violation Counting and Logging
- Each confirmed violation increments a counter.
- Violations can optionally be:
  - Logged with frame numbers or timestamps
  - Saved as annotated frames or output videos

---

### 🔁 Pipeline Summary

```text
Input Video
   ↓
YOLOv8 Object Detection
   ↓
Traffic Signal State Identification
   ↓
Stop Line Crossing Check
   ↓
Rule-Based Violation Logic
   ↓
Violation Count & Visual Output
