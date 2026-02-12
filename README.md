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

For environments without GPU, the system implements a comprehensive suite of optimizations to achieve near real-time performance on CPU-only hardware. These optimizations reduced inference time from ~500ms to ~30-50ms, representing a **10-15x speedup**.

---

### 1. Model-Level Optimizations

#### 1.1 Smaller Model Selection (yolov8n vs yolov8m)

**Problem:** YOLOv8m has 25.9M parameters and requires significant compute for each inference pass.

**Solution:** Use YOLOv8n (nano) with only 3.2M parameters — an 8x reduction in model size.

**Implementation:**
```python
# Environment variable
YOLO_MODEL=yolov8n.pt  # Instead of yolov8m.pt
```

**Trade-off:** ~5-10% lower mAP accuracy, but 6x faster inference. For surveillance use cases detecting large objects (phones, laptops), this accuracy loss is negligible.

**Speedup:** ~6x

---

#### 1.2 Reduced Input Resolution

**Problem:** Default YOLO input resolution is 640x640 pixels, requiring processing of 409,600 pixels per frame.

**Solution:** Reduce to 224x224 pixels (50,176 pixels) — an 8x reduction in pixel count.

**Implementation:**
```python
# Environment variable
YOLO_IMGSZ=224  # Instead of 640

# In inference.py
results = self.model(frame_bgr, verbose=False, conf=conf, imgsz=self.imgsz)
```

**Trade-off:** Reduced ability to detect small objects at distance. For webcam surveillance at typical distances (1-5 meters), 224px is sufficient for detecting electronics.

**Speedup:** ~2-4x

---

### 2. Runtime/Backend Optimizations

#### 2.1 ONNX Runtime Export

**Problem:** PyTorch runtime has overhead from dynamic graph execution and Python GIL limitations.

**Solution:** Export model to ONNX format and use ONNX Runtime with C++ backend for inference.

**Implementation:**
```python
# Export YOLO to ONNX
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=224, simplify=True, opset=12)

# ONNX Runtime session with optimizations
import onnxruntime as ort
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = os.cpu_count()
sess_options.inter_op_num_threads = os.cpu_count()
sess_options.enable_cpu_mem_arena = True
sess_options.enable_mem_pattern = True
sess_options.enable_mem_reuse = True

session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
```

**Key optimizations enabled:**
- `ORT_ENABLE_ALL`: Enables all graph optimizations (constant folding, node fusion, etc.)
- `intra_op_num_threads`: Parallelizes operations within a single node across CPU cores
- `inter_op_num_threads`: Parallelizes independent nodes in the graph
- `enable_cpu_mem_arena`: Pre-allocates memory to avoid allocation overhead
- `enable_mem_pattern`: Reuses memory patterns across inference calls
- `enable_mem_reuse`: Shares memory between non-overlapping tensors

**Speedup:** ~2-3x

---

#### 2.2 Intel OpenVINO Backend

**Problem:** Generic ONNX Runtime doesn't leverage Intel-specific CPU instructions (AVX-512, VNNI).

**Solution:** Use Intel OpenVINO toolkit which compiles models specifically for Intel hardware.

**Implementation:**
```python
from openvino import Core

core = Core()
model = core.read_model("yolov8n_openvino_model/yolov8n.xml")

# Optimize for low latency
config = {
    "PERFORMANCE_HINT": "LATENCY",      # Optimize for single-stream latency
    "NUM_STREAMS": "1",                  # Single inference stream
    "INFERENCE_NUM_THREADS": str(os.cpu_count()),  # Use all CPU cores
}

compiled_model = core.compile_model(model, "CPU", config)
```

**Intel-specific optimizations:**
- Automatic vectorization using AVX-512 instructions
- INT8 inference with VNNI (Vector Neural Network Instructions) on supported CPUs
- Layer fusion (Conv+BN+ReLU merged into single operation)
- Memory layout optimization for cache efficiency

**Speedup:** ~3-4x on Intel CPUs

---

#### 2.3 INT8 Dynamic Quantization

**Problem:** FP32 (32-bit floating point) operations are computationally expensive.

**Solution:** Quantize model weights to INT8 (8-bit integers), reducing memory bandwidth and enabling faster integer arithmetic.

**Implementation:**
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "yolov8n.onnx",           # Input FP32 model
    "yolov8n_int8.onnx",      # Output INT8 model
    weight_type=QuantType.QUInt8,  # Unsigned 8-bit integers
    optimize_model=True,       # Apply additional optimizations
)
```

**How it works:**
- Weights are quantized from FP32 (4 bytes) to INT8 (1 byte) — 4x memory reduction
- Activations remain FP32 but are dynamically quantized during inference
- Integer matrix multiplication is 2-4x faster than floating point on most CPUs

**Trade-off:** ~1-2% accuracy loss due to quantization error. Negligible for object detection.

**Speedup:** ~2x

---

### 3. Pipeline-Level Optimizations

#### 3.1 Frame Skipping with Result Caching

**Problem:** Processing every frame at 10+ FPS requires <100ms inference, which is difficult on CPU.

**Solution:** Process every Nth frame and reuse detection results for skipped frames.

**Implementation:**
```python
# Environment variable
FRAME_SKIP=3  # Process every 3rd frame

# In worker.py
last_detection_cache = {}

def should_process_frame(sequence_number: int) -> bool:
    return sequence_number % FRAME_SKIP == 0

def get_cached_detection(camera_id):
    return last_detection_cache.get(camera_id, ([], 0.0))

def cache_detection(camera_id, objects, infer_dt):
    last_detection_cache[camera_id] = (objects, infer_dt)

# Processing logic
if not should_process_frame(sequence_number):
    # Use cached result for skipped frames
    objects, infer_dt = get_cached_detection(camera_id)
else:
    # Run actual inference
    objects, infer_dt = detector.detect(frame)
    cache_detection(camera_id, objects, infer_dt)
```

**Why this works:** Objects in video don't move significantly between consecutive frames (at 10 FPS, objects move ~3-10 pixels between frames). Reusing detections for 2 frames introduces minimal visual lag.

**Effective throughput:** 3x (process 3.3 FPS, output 10 FPS)

---

#### 3.2 Asynchronous Multi-Worker Inference

**Problem:** Single-threaded inference blocks the main processing loop, causing frame drops.

**Solution:** Implement async inference with a thread pool that processes frames in parallel.

**Implementation:**
```python
import threading
from queue import Queue, Empty

class AsyncInferenceWorker:
    def __init__(self, detector, num_workers: int = 2):
        self.detector = detector
        self.input_queue = Queue(maxsize=10)
        self.output_queue = Queue(maxsize=10)
        self.workers = []
        self.running = True
        
        # Start worker threads
        for i in range(num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.workers.append(t)
    
    def _worker_loop(self):
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.1)
                frame, metadata = item
                objects, infer_dt = self.detector.detect(frame)
                self.output_queue.put((objects, infer_dt, metadata))
            except Empty:
                continue
    
    def submit(self, frame, metadata):
        """Non-blocking frame submission."""
        try:
            self.input_queue.put_nowait((frame, metadata))
            return True
        except:
            return False  # Queue full, process synchronously
    
    def get_result(self, timeout=0.01):
        """Non-blocking result retrieval."""
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None
```

**Benefits:**
- Main loop never blocks waiting for inference
- Multiple frames can be in-flight simultaneously
- Graceful degradation when queue is full (falls back to sync processing)

**Effective throughput:** ~2x with 2 workers

---

### 4. Auto-Detection of Best Backend

The system automatically selects the optimal backend based on available hardware and libraries:

```python
def _detect_best_backend():
    # Priority 1: OpenVINO (best for Intel CPUs)
    try:
        from openvino import Core
        core = Core()
        if "CPU" in core.available_devices:
            return "openvino"
    except ImportError:
        pass
    
    # Priority 2: INT8 quantization
    try:
        from onnxruntime.quantization import quantize_dynamic
        return "int8"
    except ImportError:
        pass
    
    # Priority 3: Basic ONNX Runtime
    try:
        import onnxruntime
        return "onnx"
    except ImportError:
        pass
    
    # Fallback: Standard YOLO
    return "yolo"
```

---

### 5. Summary of Combined Optimizations

| Optimization | Individual Speedup | Cumulative Effect |
|--------------|-------------------|-------------------|
| yolov8n model | 6x | 6x |
| 224px resolution | 2-4x | 12-24x |
| ONNX Runtime | 2-3x | 24-72x |
| OpenVINO (Intel) | 1.5x additional | 36-108x |
| INT8 Quantization | 2x | 48-144x |
| Frame Skipping (3x) | 3x throughput | 144-432x throughput |
| Async Workers (2x) | 2x throughput | 288-864x throughput |

**Realistic combined effect:** ~500ms baseline → ~30-50ms = **10-15x latency reduction**, with **3-6x throughput increase** from pipeline optimizations.

---

### Inference Backends Configuration

| Backend | Speedup | Best For | Environment Variable |
|---------|---------|----------|---------------------|
| `openvino` | 3-4x | Intel CPUs | `INFERENCE_BACKEND=openvino` |
| `int8` | 2x | Any CPU | `INFERENCE_BACKEND=int8` |
| `onnx` | 2-3x | Any CPU | `INFERENCE_BACKEND=onnx` |
| `yolo` | 1x | Baseline | `INFERENCE_BACKEND=yolo` |
| `auto` | Best | Default | `INFERENCE_BACKEND=auto` |

### CPU Optimization Config

```bash
# Environment variables
INFERENCE_BACKEND=auto     # auto, openvino, int8, onnx, yolo
YOLO_MODEL=yolov8n.pt      # Smaller model
YOLO_IMGSZ=224             # Reduced resolution
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
