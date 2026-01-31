import os
import time
import base64
import cv2
import numpy as np
from pathlib import Path

from confluent_kafka import KafkaError
from app.kafka_io import build_consumer, build_producer, headers_to_dict, produce_json
from app.inference import YoloDetector, ELECTRONICS
from app.metrics import (
    FRAMES_CONSUMED, FRAMES_PROCESSED, FRAMES_DLQ, INFERENCE_SECONDS, start_metrics_server
)

# Clip storage config
CLIPS_DIR = Path("/app/clips")
CLIPS_DIR.mkdir(exist_ok=True)
MAX_CLIPS_PER_CAMERA = int(os.getenv("MAX_CLIPS_PER_CAMERA", "500"))
CLIP_CLEANUP_INTERVAL = 100  # Check every N frames

# Alert cooldown
last_alert_time = {}
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "10"))

# Only save clip on alert (not every electronics frame)
last_clip_time = {}
CLIP_COOLDOWN_SEC = int(os.getenv("CLIP_COOLDOWN_SEC", "5"))

frame_counter = 0

def cleanup_old_clips(camera_id):
    """Delete oldest clips if over limit."""
    cam_dir = CLIPS_DIR / camera_id
    if not cam_dir.exists():
        return
    
    clips = sorted(cam_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime)
    if len(clips) > MAX_CLIPS_PER_CAMERA:
        to_delete = len(clips) - MAX_CLIPS_PER_CAMERA
        for clip in clips[:to_delete]:
            try:
                clip.unlink()
            except:
                pass
        print(f"[cleanup] Deleted {to_delete} old clips for {camera_id}")

def draw_boxes_and_encode(frame, objects):
    """Draw bounding boxes on frame and return base64 JPEG."""
    h, w = frame.shape[:2]
    for obj in objects:
        bbox = obj.get("bbox_xyxy", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (0, 0, 255) if obj.get("is_electronic") else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{obj.get('label', '')} {int(obj.get('conf', 0) * 100)}%"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(jpg.tobytes()).decode("utf-8")


def should_save_clip(camera_id):
    """Check if we should save a clip (cooldown to avoid too many)."""
    now = time.time()
    last = last_clip_time.get(camera_id, 0)
    if now - last > CLIP_COOLDOWN_SEC:
        last_clip_time[camera_id] = now
        return True
    return False

def save_clip_frame(camera_id, frame, timestamp_ms):
    """Save frame to clip directory."""
    cam_dir = CLIPS_DIR / camera_id
    cam_dir.mkdir(exist_ok=True)
    filename = f"{timestamp_ms}.jpg"
    filepath = cam_dir / filename
    cv2.imwrite(str(filepath), frame)
    return str(filepath)

def should_alert(camera_id):
    """Check if we should send an alert (cooldown logic)."""
    now = time.time()
    last = last_alert_time.get(camera_id, 0)
    if now - last > ALERT_COOLDOWN_SEC:
        last_alert_time[camera_id] = now
        return True
    return False

def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

def main():
    global frame_counter
    
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:19092")
    raw_topic = os.getenv("RAW_FRAMES_TOPIC", "raw-frames")
    det_topic = os.getenv("DETECTIONS_TOPIC", "detections")
    alerts_topic = os.getenv("ALERTS_TOPIC", "alerts")
    dlq_topic = os.getenv("DLQ_TOPIC", "raw-frames-dlq")
    group_id = os.getenv("GROUP_ID", "ai-group")
    metrics_port = int(os.getenv("METRICS_PORT", "9101"))
    model_path = os.getenv("YOLO_MODEL", "yolov8n.pt")

    start_metrics_server(metrics_port)

    consumer = build_consumer(kafka_bootstrap, group_id)
    producer = build_producer(kafka_bootstrap)
    consumer.subscribe([raw_topic])

    detector = YoloDetector(model_path)

    print(f"[worker] consuming {raw_topic}, producing to {det_topic}")
    print(f"[worker] Max clips per camera: {MAX_CLIPS_PER_CAMERA}, clip cooldown: {CLIP_COOLDOWN_SEC}s")

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            producer.poll(0)
            continue

        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                print(f"[worker] kafka error: {msg.error()}")
            continue

        FRAMES_CONSUMED.inc()
        frame_counter += 1

        h = headers_to_dict(msg.headers())
        camera_id = h.get("camera_id", "unknown")
        frame_id = h.get("frame_id", "")
        timestamp_ms = safe_int(h.get("timestamp_ms", "0"))
        sequence_number = safe_int(h.get("sequence_number", "0"))
        encoding = h.get("encoding", "jpeg")

        # Periodic cleanup
        if frame_counter % CLIP_CLEANUP_INTERVAL == 0:
            cleanup_old_clips(camera_id)

        try:
            raw = msg.value()
            if not raw or len(raw) < 10:
                raise ValueError("empty or too small frame bytes")

            if encoding.lower() != "jpeg":
                raise ValueError(f"unsupported encoding: {encoding}")

            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("cv2.imdecode failed")

            objects, infer_dt = detector.detect(frame)
            INFERENCE_SECONDS.observe(infer_dt)

            electronics_found = [o for o in objects if o.get("is_electronic")]
            
            clip_path = None
            if electronics_found and should_save_clip(camera_id):
                clip_path = save_clip_frame(camera_id, frame, timestamp_ms)
                
                if should_alert(camera_id):
                    alert_payload = {
                        "camera_id": camera_id,
                        "frame_id": frame_id,
                        "timestamp_ms": timestamp_ms,
                        "alert_type": "electronics_detected",
                        "objects": electronics_found,
                        "clip_path": clip_path,
                    }
                    produce_json(producer, alerts_topic, camera_id, alert_payload,
                                {"camera_id": camera_id, "alert_type": "electronics_detected"})
                    print(f"[ALERT] Electronics: {[o['label'] for o in electronics_found]}")

            image_data = draw_boxes_and_encode(frame.copy(), objects)

            payload = {
                "camera_id": camera_id,
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "sequence_number": sequence_number,
                "objects": objects,
                "image_data": image_data,
                "has_electronics": len(electronics_found) > 0,
                "clip_path": clip_path,
            }

            produce_json(producer, det_topic, camera_id, payload,
                        {"camera_id": camera_id, "frame_id": frame_id})
            producer.flush(2.0)

            consumer.commit(msg, asynchronous=False)
            FRAMES_PROCESSED.inc()

        except Exception as e:
            FRAMES_DLQ.inc()
            dlq_payload = {
                "error": str(e),
                "camera_id": camera_id,
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "sequence_number": sequence_number,
                "encoding": encoding,
            }
            produce_json(producer, dlq_topic, camera_id, dlq_payload,
                        {"camera_id": camera_id, "frame_id": frame_id})
            producer.flush(2.0)
            consumer.commit(msg, asynchronous=False)

        time.sleep(0.001)

if __name__ == "__main__":
    main()
