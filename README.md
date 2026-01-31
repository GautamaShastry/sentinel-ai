# Sentinel AI - Real-Time Video Surveillance System

Real-time video analytics pipeline with AI-powered object detection, alerts for electronics, and GPU acceleration support.

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐     ┌────────────┐
│   Webcam    │────►│ Ingestor │────►│   Kafka     │────►│ AI Worker  │
│   Client    │     │   (Go)   │     │ (Redpanda)  │     │  (YOLO)    │
└─────────────┘     └──────────┘     └─────────────┘     └─────┬──────┘
                                                               │
┌─────────────┐     ┌──────────┐     ┌─────────────┐           │
│  Frontend   │◄────│Result API│◄────│  Postgres   │◄──────────┘
│   (React)   │     │ (FastAPI)│     │             │
└─────────────┘     └──────────┘     └─────────────┘
```

## Features

- **Real-time object detection** using YOLOv8
- **Electronics alerts** - Detects phones, laptops, TVs, etc. and triggers alerts
- **Video clip storage** - Saves frames when electronics detected
- **Analytics dashboard** - Object counts, timeline, detection history
- **GPU acceleration** - Run on HPC clusters for 50x speedup
- **Auto-cleanup** - Prevents database/storage explosion

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

## GPU Acceleration (HPC/Hopper)

For much faster inference, run the AI worker on a GPU cluster.

### Setup

1. **On your PC** - Keep services running, stop local AI worker:
   ```bash
   make up
   docker stop infra-ai-worker-1
   ```

2. **SSH to cluster with tunnel:**
   ```bash
   ssh -R 19092:localhost:19092 username@hopper.orc.gmu.edu
   ```

3. **On cluster:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sentinel-ai.git
   cd sentinel-ai/ai-worker-python
   
   # Load modules and setup
   module load python/3.8.6-ff
   module load cuda
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Submit job
   sbatch run_hopper.sh
   ```

4. **Back on PC** - Run webcam client:
   ```bash
   cd client-webcam
   python client.py
   ```

### Slurm Job Script

```bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | yolov8n.pt | Model: n/s/m/l/x |
| `FPS` | 10 | Webcam frame rate |
| `ALERT_COOLDOWN_SEC` | 10 | Seconds between alerts |
| `CLIP_COOLDOWN_SEC` | 5 | Seconds between clip saves |
| `MAX_CLIPS_PER_CAMERA` | 500 | Max stored clips |
| `DETECTION_RETENTION_HOURS` | 24 | DB cleanup interval |

## Project Structure

```
sentinel-ai/
├── client-webcam/       # Python webcam client
├── ingestor-go/         # Go gRPC server
├── ai-worker-python/    # YOLO inference worker
├── result-api/          # FastAPI backend
├── frontend/            # React dashboard
├── infra/               # Docker Compose, Prometheus, Grafana
└── proto/               # Protobuf definitions
```

## Performance Benchmarks

| Setup | Model | Inference | FPS | Notes |
|-------|-------|-----------|-----|-------|
| Local CPU | yolov8m | ~500ms | ~2 | MacBook/PC |
| Hopper GPU (A100) | yolov8m | 16.1ms | 24.3 | GMU HPC cluster |

**Batch Processing Test (Hopper A100):**
- Video: 1920x1080 @ 29 FPS, 293 frames
- Total time: 12.1s
- Electronics detected: 117 frames

## Tech Stack

- **Ingestor**: Go, gRPC, Kafka producer
- **Message Queue**: Redpanda (Kafka-compatible)
- **AI Worker**: Python, YOLOv8, OpenCV
- **API**: FastAPI, PostgreSQL
- **Frontend**: React, Vite
- **Monitoring**: Prometheus, Grafana
