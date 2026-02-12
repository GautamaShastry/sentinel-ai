# Sentinel AI - Real-Time Video Surveillance System

Sentinel AI is a distributed real-time video analytics platform that performs object detection on live camera feeds using YOLOv8, with specialized alerting for electronics detection. The system processes 1080p video streams at 24.3 FPS with 16.1ms inference latency on GPU, representing a 30x performance improvement over CPU-based inference.

## Architecture Overview

The architecture follows an event-driven microservices pattern. A Python webcam client captures frames and streams them via gRPC to a Go-based ingestor service. The ingestor serializes frames using Protocol Buffers and publishes them to Kafka (Redpanda) for decoupled, fault-tolerant message delivery. Python AI workers consume frames from Kafka, run YOLOv8 inference, and persist detection results to PostgreSQL. A FastAPI backend exposes REST endpoints for detection history, alerts, and analytics. A React frontend provides a real-time dashboard with object counts, detection timelines, and alert management.

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐     ┌────────────┐
│   Webcam    │────►│ Ingestor │────►│   Kafka     │────►│ AI Worker  │
│   Client    │gRPC │   (Go)   │ PB  │ (Redpanda)  │     │  (YOLOv8)  │
└─────────────┘     └──────────┘     └─────────────┘     └─────┬──────┘
                                                               │
┌─────────────┐     ┌──────────┐     ┌─────────────┐           │
│  Frontend   │◄────│Result API│◄────│  Postgres   │◄──────────┘
│   (React)   │ WS  │ (FastAPI)│REST │             │
└─────────────┘     └──────────┘     └─────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Prometheus  ──►  Grafana       │
│  (Metrics)       (Dashboards)   │
└─────────────────────────────────┘
```

## Performance Benchmarks

| Setup | Model | Inference | FPS | Notes |
|-------|-------|-----------|-----|-------|
| Local CPU (optimized) | yolov8n + ONNX | ~50-80ms | 12-20 | With all CPU optimizations |
| Local CPU (baseline) | yolov8m | ~500ms | ~2 | MacBook/PC |
| Hopper GPU (A100) | yolov8m | 16.1ms | 24.3 | GMU HPC cluster |

**Batch Processing Test (Hopper A100):**
- Video: 1920x1080 @ 29 FPS, 293 frames
- Total processing time: 12.1s
- Electronics detected: 117 frames
- **30x speedup** over CPU-based inference

## CPU Optimization Features

For environments without GPU, the system includes multiple optimizations to achieve real-time performance:

### Inference Backends

| Backend | Speedup | Best For | Description |
|---------|---------|----------|-------------|
| `openvino` | 3-4x | Intel CPUs | Intel OpenVINO with hardware-specific optimizations |
| `int8` | 2x | Any CPU | INT8 quantized ONNX (~2x faster, minimal accuracy loss) |
| `onnx` | 2-3x | Any CPU | ONNX Runtime with multi-threaded execution |
| `yolo` | 1x | Baseline | Standard ultralytics (most compatible) |
| `auto` | Best | Default | Auto-detects best available backend |

### Additional Optimizations

| Optimization | Speedup | Description |
|--------------|---------|-------------|
| Smaller Model | 6x | yolov8n vs yolov8m with minor accuracy tradeoff |
| Reduced Resolution | 2-4x | imgsz=224 instead of 640 |
| Frame Skipping | 3x | Process every Nth frame, cache results |
| Async Workers | 2x | Multi-threaded inference pipeline |

**Combined effect:** ~30-50ms inference on Intel CPU with OpenVINO (vs ~500ms baseline) = **10-15x speedup**

### CPU Optimization Config

```bash
# Environment variables
INFERENCE_BACKEND=auto     # auto, openvino, int8, onnx, yolo
YOLO_IMGSZ=224             # Smaller input resolution
FRAME_SKIP=3               # Process every 3rd frame
ASYNC_PROCESSING=true      # Enable async inference
NUM_INFERENCE_WORKERS=2    # Parallel inference threads
```

## Key Features

- **Real-time Object Detection**: YOLOv8 inference with configurable models (yolov8n through yolov8x) to balance accuracy and throughput
- **Multi-Backend Inference**: OpenVINO (Intel), INT8 quantization, ONNX Runtime with auto-detection of best backend
- **CPU-Optimized Inference**: Frame skipping, async workers, reduced resolution for 10-15x CPU speedup without GPU
- **Electronics Alerting**: Specialized detection for phones, laptops, TVs, keyboards, and other electronics with configurable alert cooldowns
- **GPU Acceleration**: HPC cluster deployment via SLURM with NVIDIA A100 support for production-scale inference
- **Event-Driven Streaming**: Kafka-based message queue for decoupled, fault-tolerant frame delivery between services
- **Production Observability**: Prometheus metrics collection and Grafana dashboards for monitoring inference latency, throughput, and detection rates
- **Auto-Cleanup Policies**: Automatic retention policies prevent unbounded database/storage growth
- **Hybrid Deployment**: GPU inference on remote clusters while other services run locally, connected via SSH tunneling

## Technical Decisions

- **OpenVINO** for 3-4x faster inference on Intel CPUs with hardware-specific optimizations
- **INT8 Quantization** for ~2x speedup with minimal accuracy loss on any CPU
- **ONNX Runtime** for 2-3x faster CPU inference with multi-threaded execution
- **Redpanda** as Kafka-compatible message broker for lower operational overhead
- **Protocol Buffers** for efficient binary serialization of video frames
- **opencv-python-headless** for server-side image processing without GUI dependencies
- **Frame skipping with result caching** for throughput optimization on CPU
- **Async inference pipeline** with configurable worker threads
- **Docker Compose** for single-command deployment of all services
- **Horizontal scaling** support for AI workers

## Quick Start (Local)

```bash
# Start all services
make up

# Run webcam client
cd client-webcam
pip install -r requirements.txt
python client.py
```

**Access:**
- Frontend: http://localhost:3000
- Grafana: http://localhost:3001 (admin/sentinel)
- Prometheus: http://localhost:9090
- API: http://localhost:8080

## GPU Acceleration (HPC/Hopper)

For GPU-accelerated batch processing on HPC clusters:

### Batch Processing Mode

Bypasses Kafka dependencies for standalone video file analysis on GPU nodes.

```bash
# SSH to cluster
ssh username@hopper.orc.gmu.edu

# Setup (first time)
cd sentinel-ai/ai-worker-python
module load python/3.8.6-ff
module load cuda
python -m venv venv
source venv/bin/activate
pip install -r requirements-batch.txt

# Submit batch job
sbatch run_batch.sh /path/to/video.mp4
```

### SLURM Configuration

```bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
```

### Real-Time Mode (SSH Tunneling)

For real-time inference with local services:

```bash
# On your PC - Start services, stop local AI worker
make up
docker stop infra-ai-worker-1

# SSH with reverse tunnel
ssh -R 19092:localhost:19092 username@hopper.orc.gmu.edu

# On cluster - Submit streaming job
sbatch run_hopper.sh
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/cameras` | List active cameras |
| `/latest?camera_id=X` | Latest detection with image |
| `/detections?camera_id=X` | Detection history |
| `/alerts` | All alerts |
| `/alerts/recent` | Recent alerts (fast) |
| `/analytics/summary?camera_id=X` | Detection stats |
| `/analytics/electronics` | Electronics detection stats |
| `/analytics/timeline?camera_id=X` | Time-series data |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | yolov8n.pt | Model: n/s/m/l/x |
| `YOLO_IMGSZ` | 224 | Input resolution (lower = faster) |
| `USE_ONNX` | true | Enable ONNX Runtime for CPU |
| `FRAME_SKIP` | 3 | Process every Nth frame |
| `ASYNC_PROCESSING` | true | Enable async inference |
| `NUM_INFERENCE_WORKERS` | 2 | Parallel inference threads |
| `FPS` | 10 | Webcam frame rate |
| `ALERT_COOLDOWN_SEC` | 10 | Seconds between alerts |
| `CLIP_COOLDOWN_SEC` | 5 | Seconds between clip saves |
| `MAX_CLIPS_PER_CAMERA` | 500 | Max stored clips |
| `DETECTION_RETENTION_HOURS` | 24 | DB cleanup interval |

## Project Structure

```
sentinel-ai/
├── client-webcam/       # Python webcam client (gRPC)
├── ingestor-go/         # Go gRPC server + Kafka producer
├── ai-worker-python/    # YOLOv8 inference worker
├── result-api/          # FastAPI REST backend
├── frontend/            # React dashboard (Vite)
├── infra/               # Docker Compose, Prometheus, Grafana
└── proto/               # Protocol Buffer definitions
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Video Ingestion | Go, gRPC, Protocol Buffers |
| Message Queue | Redpanda (Kafka-compatible) |
| ML Inference | Python, YOLOv8, ONNX Runtime, OpenCV |
| Backend API | FastAPI, PostgreSQL |
| Frontend | React, Vite |
| Monitoring | Prometheus, Grafana |
| GPU Compute | SLURM, NVIDIA A100 |
| Containerization | Docker Compose |
