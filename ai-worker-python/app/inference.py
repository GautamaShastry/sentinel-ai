import os
import time
from pathlib import Path

import cv2
import numpy as np

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
    """Standard YOLO detector using ultralytics."""
    
    def __init__(self, model_path: str = "yolov8n.pt", imgsz: int = 224):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        print(f"[YoloDetector] Loaded {model_path} with imgsz={imgsz}")

    def detect(self, frame_bgr, conf: float = 0.25):
        t0 = time.time()
        results = self.model(frame_bgr, verbose=False, conf=conf, imgsz=self.imgsz)
        dt = time.time() - t0
        return self._parse_results(results), dt

    def _parse_results(self, results):
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
        return objects


class ONNXDetector:
    """Optimized ONNX Runtime detector for faster CPU inference."""
    
    def __init__(self, model_path: str = "yolov8n.onnx", imgsz: int = 224, use_int8: bool = False):
        import onnxruntime as ort
        
        # Use all available CPU optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count()
        sess_options.inter_op_num_threads = os.cpu_count()
        
        # Enable memory optimizations
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.use_int8 = use_int8
        print(f"[ONNXDetector] Loaded {model_path} with imgsz={imgsz}, threads={os.cpu_count()}")

    def detect(self, frame_bgr, conf: float = 0.25):
        t0 = time.time()
        
        # Preprocess
        img = cv2.resize(frame_bgr, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # Add batch dim
        
        # Inference
        outputs = self.session.run(None, {self.input_name: img})
        
        # Parse YOLO output
        objects = self._parse_yolo_output(outputs[0], frame_bgr.shape, conf)
        
        dt = time.time() - t0
        return objects, dt

    def _parse_yolo_output(self, output, orig_shape, conf_thresh):
        """Parse YOLOv8 ONNX output format."""
        objects = []
        
        # YOLOv8 output: [1, 84, 8400] -> transpose to [8400, 84]
        predictions = output[0].T
        
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.imgsz
        scale_y = orig_h / self.imgsz
        
        for pred in predictions:
            cx, cy, w, h = pred[:4]
            class_scores = pred[4:]
            
            cls_id = np.argmax(class_scores)
            conf = class_scores[cls_id]
            
            if conf < conf_thresh:
                continue
            
            x1 = (cx - w / 2) * scale_x
            y1 = (cy - h / 2) * scale_y
            x2 = (cx + w / 2) * scale_x
            y2 = (cy + h / 2) * scale_y
            
            label = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"class_{cls_id}"
            is_electronic = label in ELECTRONICS
            
            objects.append({
                "cls_id": int(cls_id),
                "label": label,
                "conf": float(conf),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "is_electronic": is_electronic,
            })
        
        objects = self._nms(objects, iou_thresh=0.5)
        return objects

    def _nms(self, objects, iou_thresh=0.5):
        if not objects:
            return []
        objects = sorted(objects, key=lambda x: x['conf'], reverse=True)
        keep = []
        while objects:
            best = objects.pop(0)
            keep.append(best)
            objects = [obj for obj in objects if self._iou(best['bbox_xyxy'], obj['bbox_xyxy']) < iou_thresh]
        return keep

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)


class OpenVINODetector:
    """Intel OpenVINO optimized detector - 2-3x faster on Intel CPUs."""
    
    def __init__(self, model_path: str = "yolov8n_openvino_model", imgsz: int = 224):
        from openvino import Core
        
        self.core = Core()
        self.imgsz = imgsz
        
        # Load model
        model = self.core.read_model(model_path + "/yolov8n.xml")
        
        # Optimize for throughput on CPU
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": str(os.cpu_count()),
        }
        
        self.compiled_model = self.core.compile_model(model, "CPU", config)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        print(f"[OpenVINODetector] Loaded {model_path} with imgsz={imgsz}, threads={os.cpu_count()}")

    def detect(self, frame_bgr, conf: float = 0.25):
        t0 = time.time()
        
        # Preprocess
        img = cv2.resize(frame_bgr, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # Inference
        result = self.compiled_model([img])[self.output_layer]
        
        # Parse output
        objects = self._parse_yolo_output(result, frame_bgr.shape, conf)
        
        dt = time.time() - t0
        return objects, dt

    def _parse_yolo_output(self, output, orig_shape, conf_thresh):
        objects = []
        predictions = output[0].T
        
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.imgsz
        scale_y = orig_h / self.imgsz
        
        for pred in predictions:
            cx, cy, w, h = pred[:4]
            class_scores = pred[4:]
            
            cls_id = np.argmax(class_scores)
            conf = class_scores[cls_id]
            
            if conf < conf_thresh:
                continue
            
            x1 = (cx - w / 2) * scale_x
            y1 = (cy - h / 2) * scale_y
            x2 = (cx + w / 2) * scale_x
            y2 = (cy + h / 2) * scale_y
            
            label = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"class_{cls_id}"
            is_electronic = label in ELECTRONICS
            
            objects.append({
                "cls_id": int(cls_id),
                "label": label,
                "conf": float(conf),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "is_electronic": is_electronic,
            })
        
        return self._nms(objects, iou_thresh=0.5)

    def _nms(self, objects, iou_thresh=0.5):
        if not objects:
            return []
        objects = sorted(objects, key=lambda x: x['conf'], reverse=True)
        keep = []
        while objects:
            best = objects.pop(0)
            keep.append(best)
            objects = [obj for obj in objects if self._iou(best['bbox_xyxy'], obj['bbox_xyxy']) < iou_thresh]
        return keep

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)


class INT8Detector:
    """INT8 quantized ONNX detector - ~2x faster with minimal accuracy loss."""
    
    def __init__(self, model_path: str = "yolov8n_int8.onnx", imgsz: int = 224):
        import onnxruntime as ort
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count()
        sess_options.inter_op_num_threads = os.cpu_count()
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        print(f"[INT8Detector] Loaded {model_path} with imgsz={imgsz}")

    def detect(self, frame_bgr, conf: float = 0.25):
        t0 = time.time()
        
        # Preprocess - keep as uint8 for INT8 model
        img = cv2.resize(frame_bgr, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        outputs = self.session.run(None, {self.input_name: img})
        objects = self._parse_yolo_output(outputs[0], frame_bgr.shape, conf)
        
        dt = time.time() - t0
        return objects, dt

    def _parse_yolo_output(self, output, orig_shape, conf_thresh):
        objects = []
        predictions = output[0].T
        
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.imgsz
        scale_y = orig_h / self.imgsz
        
        for pred in predictions:
            cx, cy, w, h = pred[:4]
            class_scores = pred[4:]
            
            cls_id = np.argmax(class_scores)
            conf = class_scores[cls_id]
            
            if conf < conf_thresh:
                continue
            
            x1 = (cx - w / 2) * scale_x
            y1 = (cy - h / 2) * scale_y
            x2 = (cx + w / 2) * scale_x
            y2 = (cy + h / 2) * scale_y
            
            label = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"class_{cls_id}"
            is_electronic = label in ELECTRONICS
            
            objects.append({
                "cls_id": int(cls_id),
                "label": label,
                "conf": float(conf),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "is_electronic": is_electronic,
            })
        
        return self._nms(objects, iou_thresh=0.5)

    def _nms(self, objects, iou_thresh=0.5):
        if not objects:
            return []
        objects = sorted(objects, key=lambda x: x['conf'], reverse=True)
        keep = []
        while objects:
            best = objects.pop(0)
            keep.append(best)
            objects = [obj for obj in objects if self._iou(best['bbox_xyxy'], obj['bbox_xyxy']) < iou_thresh]
        return keep

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)


def export_to_onnx(model_path: str = "yolov8n.pt", imgsz: int = 224):
    """Export YOLO model to ONNX format."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    onnx_path = model_path.replace(".pt", ".onnx")
    model.export(format="onnx", imgsz=imgsz, simplify=True, opset=12)
    print(f"[export] Exported {model_path} -> {onnx_path}")
    return onnx_path


def export_to_openvino(model_path: str = "yolov8n.pt", imgsz: int = 224):
    """Export YOLO model to OpenVINO IR format."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    model.export(format="openvino", imgsz=imgsz, half=False)
    openvino_path = model_path.replace(".pt", "_openvino_model")
    print(f"[export] Exported {model_path} -> {openvino_path}")
    return openvino_path


def quantize_to_int8(onnx_path: str = "yolov8n.onnx", calibration_data_path: str = None):
    """Quantize ONNX model to INT8 for faster inference."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    int8_path = onnx_path.replace(".onnx", "_int8.onnx")
    
    # Dynamic quantization (no calibration data needed)
    quantize_dynamic(
        onnx_path,
        int8_path,
        weight_type=QuantType.QUInt8,
        optimize_model=True,
    )
    
    print(f"[quantize] Quantized {onnx_path} -> {int8_path}")
    return int8_path


def create_detector(
    backend: str = None,
    model_path: str = None,
    imgsz: int = None
):
    """
    Factory function to create the best available detector.
    
    Backends:
    - "yolo": Standard ultralytics YOLO (slowest, most compatible)
    - "onnx": ONNX Runtime (2-3x faster)
    - "openvino": Intel OpenVINO (2-3x faster on Intel CPUs)
    - "int8": INT8 quantized ONNX (~2x faster than FP32 ONNX)
    - "auto": Auto-detect best backend
    """
    
    # Defaults from environment
    if backend is None:
        backend = os.getenv("INFERENCE_BACKEND", "auto").lower()
    if model_path is None:
        model_path = os.getenv("YOLO_MODEL", "yolov8n.pt")
    if imgsz is None:
        imgsz = int(os.getenv("YOLO_IMGSZ", "224"))
    
    # Auto-detect best backend
    if backend == "auto":
        backend = _detect_best_backend()
    
    print(f"[detector] Using backend: {backend}")
    
    if backend == "openvino":
        openvino_path = model_path.replace(".pt", "_openvino_model")
        if not Path(openvino_path).exists():
            print(f"[detector] OpenVINO model not found, exporting...")
            export_to_openvino(model_path, imgsz)
        return OpenVINODetector(openvino_path, imgsz)
    
    elif backend == "int8":
        onnx_path = model_path.replace(".pt", ".onnx")
        int8_path = model_path.replace(".pt", "_int8.onnx")
        
        if not Path(int8_path).exists():
            if not Path(onnx_path).exists():
                print(f"[detector] ONNX model not found, exporting...")
                export_to_onnx(model_path, imgsz)
            print(f"[detector] INT8 model not found, quantizing...")
            quantize_to_int8(onnx_path)
        
        return INT8Detector(int8_path, imgsz)
    
    elif backend == "onnx":
        onnx_path = model_path.replace(".pt", ".onnx")
        if not Path(onnx_path).exists():
            print(f"[detector] ONNX model not found, exporting...")
            export_to_onnx(model_path, imgsz)
        return ONNXDetector(onnx_path, imgsz)
    
    else:  # "yolo" or fallback
        return YoloDetector(model_path, imgsz)


def _detect_best_backend():
    """Auto-detect the best available backend."""
    
    # Check for OpenVINO (best for Intel CPUs)
    try:
        from openvino import Core
        core = Core()
        devices = core.available_devices
        if "CPU" in devices:
            print("[auto-detect] OpenVINO available - using for Intel CPU optimization")
            return "openvino"
    except ImportError:
        pass
    
    # Check for ONNX Runtime with quantization support
    try:
        import onnxruntime
        from onnxruntime.quantization import quantize_dynamic
        print("[auto-detect] ONNX Runtime with INT8 quantization available")
        return "int8"
    except ImportError:
        pass
    
    # Check for basic ONNX Runtime
    try:
        import onnxruntime
        print("[auto-detect] ONNX Runtime available")
        return "onnx"
    except ImportError:
        pass
    
    # Fallback to standard YOLO
    print("[auto-detect] Falling back to standard YOLO")
    return "yolo"
