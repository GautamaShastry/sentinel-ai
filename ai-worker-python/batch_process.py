#!/usr/bin/env python3
"""
Batch video processor for HPC/GPU environments.
Processes a video file and outputs detection results to JSON.

Usage:
    python batch_process.py input_video.mp4 --output results.json
    python batch_process.py input_video.mp4 --output-dir ./detections --save-frames
"""

import argparse
import json
import time
import os
from pathlib import Path

import cv2
import numpy as np

from app.inference import YoloDetector, ELECTRONICS


def process_video(
    video_path: str,
    output_path: str = None,
    output_dir: str = None,
    save_frames: bool = False,
    model: str = "yolov8m.pt",
    skip_frames: int = 1,
    max_frames: int = None,
):
    """Process a video file and detect objects."""
    
    print(f"Loading model: {model}")
    detector = YoloDetector(model)
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    if output_dir and save_frames:
        frames_dir = Path(output_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    frame_count = 0
    processed_count = 0
    total_inference_time = 0
    electronics_detected = 0
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames if requested
        if frame_count % skip_frames != 0:
            continue
        
        if max_frames and processed_count >= max_frames:
            break
        
        # Run detection
        objects, inference_time = detector.detect(frame)
        total_inference_time += inference_time
        processed_count += 1
        
        # Check for electronics
        electronics = [o for o in objects if o.get("is_electronic")]
        if electronics:
            electronics_detected += 1
        
        # Store result
        result = {
            "frame_number": frame_count,
            "timestamp_sec": frame_count / fps,
            "inference_time_ms": inference_time * 1000,
            "objects": objects,
            "electronics_count": len(electronics),
        }
        results.append(result)
        
        # Save frame with detections if requested
        if save_frames and output_dir:
            # Draw boxes
            for obj in objects:
                bbox = obj.get("bbox_xyxy", [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = (0, 0, 255) if obj.get("is_electronic") else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{obj.get('label', '')} {int(obj.get('conf', 0) * 100)}%"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
        
        # Progress
        if processed_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = processed_count / elapsed
            print(f"Processed {processed_count} frames ({frame_count}/{total_frames}) - {fps_actual:.1f} FPS")
    
    cap.release()
    
    elapsed = time.time() - start_time
    avg_inference = (total_inference_time / processed_count * 1000) if processed_count > 0 else 0
    
    # Summary
    summary = {
        "video_path": video_path,
        "model": model,
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "skip_frames": skip_frames,
        "elapsed_seconds": elapsed,
        "avg_fps": processed_count / elapsed if elapsed > 0 else 0,
        "avg_inference_ms": avg_inference,
        "frames_with_electronics": electronics_detected,
        "detections": results,
    }
    
    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    if output_dir:
        summary_path = Path(output_dir) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Frames processed: {processed_count}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Average FPS: {processed_count / elapsed:.1f}")
    print(f"Avg inference: {avg_inference:.1f}ms")
    print(f"Frames with electronics: {electronics_detected}")
    print(f"{'='*50}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch video processor with YOLO")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--output-dir", "-d", help="Output directory for results and frames")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO model (default: yolov8m.pt)")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    
    args = parser.parse_args()
    
    if not args.output and not args.output_dir:
        args.output = args.video.rsplit(".", 1)[0] + "_detections.json"
    
    process_video(
        video_path=args.video,
        output_path=args.output,
        output_dir=args.output_dir,
        save_frames=args.save_frames,
        model=args.model,
        skip_frames=args.skip,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
