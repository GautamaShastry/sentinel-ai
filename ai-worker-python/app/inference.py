import time
from ultralytics import YOLO

# COCO class names for YOLO
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Electronics we want to alert on
ELECTRONICS = {"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
               "oven", "toaster", "refrigerator", "clock", "hair drier"}

class YoloDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame_bgr):
        t0 = time.time()
        results = self.model(frame_bgr, verbose=False, conf=0.2, imgsz=320)
        dt = time.time() - t0

        objects = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])
                conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
                xyxy = b.xyxy[0].tolist()
                label = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"class_{cls_id}"
                is_electronic = label in ELECTRONICS
                objects.append({
                    "cls_id": cls_id,
                    "label": label,
                    "conf": conf,
                    "bbox_xyxy": [float(x) for x in xyxy],
                    "is_electronic": is_electronic,
                })

        return objects, dt
