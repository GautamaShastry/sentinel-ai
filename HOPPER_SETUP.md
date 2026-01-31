# Running Sentinel AI on GMU Hopper Cluster

This guide explains how to run the AI worker on Hopper's GPUs while keeping other services on your local PC.

## Architecture

```
Your PC                              Hopper (GPU)
────────                             ────────────
webcam client ──┐
                ├──► Kafka ◄──SSH Tunnel──► ai-worker (GPU)
frontend      ◄─┤
result-api    ◄─┘
```

## Step 1: Prepare Your Local PC

1. Make sure Docker is running and services are up:
   ```powershell
   make up
   ```

2. Verify services are running:
   ```powershell
   docker ps
   ```

3. Stop the local ai-worker (we'll use Hopper's instead):
   ```powershell
   docker stop infra-ai-worker-1
   ```

## Step 2: Push Code to GitHub

```powershell
git add .
git commit -m "Add Hopper support"
git push origin main
```

## Step 3: Connect to Hopper with SSH Tunnel

Open a terminal and SSH with reverse port forwarding:

```bash
ssh -R 19092:localhost:19092 YOUR_USERNAME@hopper.gmu.edu
```

**Keep this terminal open!** The tunnel only works while connected.

## Step 4: Setup on Hopper

In the SSH session:

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/sentinel-ai.git
cd sentinel-ai/ai-worker-python

# Make script executable
chmod +x run_hopper.sh

# Submit the job
sbatch run_hopper.sh
```

## Step 5: Monitor the Job

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f sentinel-*.out

# Check GPU usage (if job is running)
srun --jobid=YOUR_JOB_ID nvidia-smi
```

## Step 6: Run Webcam Client (on your PC)

In a new PowerShell window:

```powershell
cd client-webcam
python client.py
```

## Step 7: View Results

- Frontend: http://localhost:3000
- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090

## Troubleshooting

### "Connection refused" on Hopper
- Make sure SSH tunnel is still connected
- Verify Kafka is running: `docker logs infra-redpanda-1`

### Job pending too long
- Check queue: `squeue -p gpuq`
- Try different partition or reduce time/memory

### No detections appearing
- Check worker logs: `cat sentinel-*.out`
- Verify topics exist: `docker exec infra-redpanda-1 rpk topic list`

## Performance Comparison

| Setup | Inference Time | FPS |
|-------|---------------|-----|
| Local CPU | ~500ms | ~2 |
| Hopper GPU (yolov8m) | ~25ms | ~40 |
| Hopper GPU (yolov8x) | ~50ms | ~20 |

## Stopping

1. Cancel Hopper job:
   ```bash
   scancel YOUR_JOB_ID
   ```

2. Close SSH tunnel (Ctrl+C or close terminal)

3. Optionally restart local ai-worker:
   ```powershell
   docker start infra-ai-worker-1
   ```
