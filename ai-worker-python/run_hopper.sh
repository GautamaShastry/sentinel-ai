#!/bin/bash
#SBATCH --job-name=sentinel-ai
#SBATCH --output=sentinel-%j.out
#SBATCH --error=sentinel-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G

echo "Starting Sentinel AI Worker on GPU"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Load modules
module load python/3.11
module load cuda

# Setup virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Kafka via SSH tunnel
export KAFKA_BOOTSTRAP="localhost:19092"
export RAW_FRAMES_TOPIC="raw-frames"
export DETECTIONS_TOPIC="detections"
export ALERTS_TOPIC="alerts"
export DLQ_TOPIC="raw-frames-dlq"
export GROUP_ID="ai-group-hopper"
export METRICS_PORT="9101"
export YOLO_MODEL="yolov8m.pt"
export MAX_CLIPS_PER_CAMERA="100"
export CLIP_COOLDOWN_SEC="10"

echo "Kafka: $KAFKA_BOOTSTRAP"
echo "Model: $YOLO_MODEL"
nvidia-smi

python -m app.worker
