import os
import json
import threading
import time
from confluent_kafka import Consumer, KafkaError

from app.db import execute

DETECTIONS_TOPIC = os.getenv("DETECTIONS_TOPIC", "detections")
ALERTS_TOPIC = os.getenv("ALERTS_TOPIC", "alerts")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:19092")
GROUP_ID = os.getenv("GROUP_ID", "result-api-group")

LATEST = {}  # camera_id -> last detection payload (in-memory cache)
LATEST_ALERTS = []  # Recent alerts (in-memory, max 100)

def _build_consumer(topics):
    c = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": GROUP_ID,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    })
    c.subscribe(topics)
    return c

def _upsert_detection(payload: dict):
    execute(
        """
        INSERT INTO detections (camera_id, frame_id, timestamp_ms, sequence_number, objects, has_electronics)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s)
        ON CONFLICT (camera_id, frame_id) DO NOTHING
        """,
        (
            payload["camera_id"],
            payload["frame_id"],
            int(payload["timestamp_ms"]),
            int(payload["sequence_number"]),
            json.dumps(payload["objects"]),
            payload.get("has_electronics", False),
        ),
    )

def _insert_alert(payload: dict):
    execute(
        """
        INSERT INTO alerts (camera_id, frame_id, timestamp_ms, alert_type, objects, clip_path)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s)
        """,
        (
            payload["camera_id"],
            payload["frame_id"],
            int(payload["timestamp_ms"]),
            payload["alert_type"],
            json.dumps(payload["objects"]),
            payload.get("clip_path"),
        ),
    )


def start_consumer_loop():
    c = _build_consumer([DETECTIONS_TOPIC, ALERTS_TOPIC])
    print(f"[result-api] consuming {DETECTIONS_TOPIC}, {ALERTS_TOPIC} on {KAFKA_BOOTSTRAP}")

    while True:
        msg = c.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                print(f"[result-api] kafka error: {msg.error()}")
            continue

        try:
            payload = json.loads(msg.value().decode("utf-8"))
            topic = msg.topic()
            
            if topic == DETECTIONS_TOPIC:
                if "camera_id" not in payload:
                    raise ValueError("missing camera_id")
                LATEST[payload["camera_id"]] = payload
                _upsert_detection(payload)
                
            elif topic == ALERTS_TOPIC:
                _insert_alert(payload)
                LATEST_ALERTS.insert(0, payload)
                if len(LATEST_ALERTS) > 100:
                    LATEST_ALERTS.pop()
                print(f"[result-api] Alert: {payload.get('alert_type')} on {payload.get('camera_id')}")
                
            c.commit(msg, asynchronous=False)
        except Exception as e:
            print(f"[result-api] failed to handle msg: {e}")
            c.commit(msg, asynchronous=False)

        time.sleep(0.001)

def run_in_background():
    t = threading.Thread(target=start_consumer_loop, daemon=True)
    t.start()
