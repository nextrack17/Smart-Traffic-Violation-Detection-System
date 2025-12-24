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
