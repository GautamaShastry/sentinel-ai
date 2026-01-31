#!/bin/bash
#SBATCH --job-name=sentinel-batch
#SBATCH --output=batch-%j.out
#SBATCH --error=batch-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G

# Usage: sbatch run_batch.sh /path/to/video.mp4

VIDEO_FILE=$1

if [ -z "$VIDEO_FILE" ]; then
    echo "Usage: sbatch run_batch.sh /path/to/video.mp4"
    exit 1
fi

echo "Processing video: $VIDEO_FILE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Load modules
module load python/3.8.6-ff
module load cuda/12.6.3

# Activate venv
source venv/bin/activate

# Show GPU info
nvidia-smi

# Create output directory
OUTPUT_DIR="results_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Run batch processing
python batch_process.py "$VIDEO_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --save-frames \
    --model yolov8m.pt \
    --skip 1

echo "Results saved to: $OUTPUT_DIR"
