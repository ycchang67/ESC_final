
from typing import Dict


try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency at runtime
    YOLO = None


class DetectionResult:
    def __init__(self, bbox: Dict[str, float], confidence: float, source: str) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.source = source


class PersonDetector:
    """YOLO person detector with fallback central crop box."""

    def __init__(self, model_name: str = "yolov8n.pt", use_yolo: bool = True) -> None:
        self.model_name = model_name
        self.model = YOLO(model_name) if (YOLO is not None and use_yolo) else None

    def detect_largest_person(self, image_rgb) -> DetectionResult:
        if self.model is None:
            return self._fallback(image_rgb)

        results = self.model.predict(source=image_rgb, verbose=False)
        if not results:
            return self._fallback(image_rgb)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return self._fallback(image_rgb)

        best = None
        best_area = -1.0
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id != 0:  # COCO class 0 = person
                continue

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            area = max(1.0, (x2 - x1) * (y2 - y1))
            if area > best_area:
                best_area = area
                best = {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(boxes.conf[i].item()),
                }

        if best is None:
            return self._fallback(image_rgb)

        bbox = {k: best[k] for k in ("x1", "y1", "x2", "y2")}
        return DetectionResult(bbox=bbox, confidence=best["confidence"], source="yolo")

    def _fallback(self, image_rgb) -> DetectionResult:
        h, w = image_rgb.shape[:2]
        x1, y1 = 0.2 * w, 0.1 * h
        x2, y2 = 0.8 * w, 0.95 * h
        return DetectionResult(
            bbox={"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
            confidence=0.0,
            source="fallback",
        )
