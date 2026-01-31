package server

import (
	"context"
	"io"
	"log"
	"strconv"
	"time"

	"sentinel-ai/ingestor-go/internal/kafka"
	"sentinel-ai/ingestor-go/internal/metrics"
	"sentinel-ai/ingestor-go/pb"

	"google.golang.org/grpc"
)

type grpcSrv struct {
	pb.UnimplementedVideoStreamerServer
	producer *kafka.Producer
	met      *metrics.Metrics
}

func Register(s *grpc.Server, producer *kafka.Producer, met *metrics.Metrics) {
	pb.RegisterVideoStreamerServer(s, &grpcSrv{
		producer: producer,
		met:      met,
	})
}


func (g *grpcSrv) UploadVideo(stream pb.VideoStreamer_UploadVideoServer) error {
	var frames int64
	start := time.Now()

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			elapsed := time.Since(start).Seconds()
			var fps int32 = 0
			if elapsed > 0 {
				fps = int32(float64(frames) / elapsed)
			}
			_ = fps // could use for logging
			return stream.SendAndClose(&pb.UploadStatus{
				Message:        "stream complete",
				Success:        true,
				RecommendedFps: 30,
			})
		}
		if err != nil {
			log.Printf("recv error: %v", err)
			return err
		}

		// Basic validation
		if req.CameraId == "" || req.FrameId == "" || len(req.ImageData) == 0 {
			continue
		}

		// Metrics
		g.met.FramesIngested.Inc()
		g.met.BytesIngested.Add(float64(len(req.ImageData)))
		frames++

		// Keep per-camera ordering by using camera_id as key
		key := []byte(req.CameraId)

		headers := map[string]string{
			"camera_id":       req.CameraId,
			"frame_id":        req.FrameId,
			"timestamp_ms":    strconv.FormatInt(req.TimestampMs, 10),
			"sequence_number": strconv.FormatInt(req.SequenceNumber, 10),
			"encoding":        req.Encoding,
		}

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		err = g.producer.ProduceRawFrame(ctx, key, req.ImageData, headers)
		cancel()

		if err != nil {
			log.Printf("produce failed: %v", err)
		}
	}
}
