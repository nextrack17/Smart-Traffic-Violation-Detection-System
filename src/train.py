from ultralytics import YOLO
from pathlib import Path

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    model = YOLO("yolov8s.pt")

    model.train(
        data=str(BASE_DIR / "data" / "data.yaml"),
        epochs=30,
        imgsz=640,
        batch=8,
        project=str(BASE_DIR / "models"),
        name="traffic_violation_model",
        workers=0,      # IMPORTANT for Windows
        device=0        # GPU (if available)
    )

if __name__ == "__main__":
    main()
