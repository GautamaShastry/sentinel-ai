# Sentinel AI (gRPC + Redpanda + YOLO + FastAPI + Postgres)

A real-time video analytics pipeline:
- gRPC streaming ingestion (Go)
- Buffering + scaling via Redpanda (Kafka API)
- YOLO inference worker (Python) with DLQ
- Results API (FastAPI) persisting to Postgres + live SSE stream
- Prometheus metrics

## Run

1) Generate protobufs
```bash
make gen
```

2) Start stack
```bash
make up
```

3) Start webcam client (local machine)
```bash
cd client-webcam
pip install -r requirements.txt
export GRPC_TARGET=localhost:50051
export CAMERA_ID=camera_1
export FPS=25
python client.py
```

## Query results

- Health: http://localhost:8080/health
- Cameras: http://localhost:8080/cameras
- Latest: http://localhost:8080/latest?camera_id=camera_1
- History: http://localhost:8080/detections?camera_id=camera_1&since_ms=0&limit=25
- Live SSE stream:
```bash
curl http://localhost:8080/stream/detections?camera_id=camera_1
```

## Metrics

- Prometheus: http://localhost:9090
- Ingestor metrics: http://localhost:9100/metrics
- Worker metrics: http://localhost:9101/metrics
