package main

import (
	"context"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"google.golang.org/grpc"

	"sentinel-ai/ingestor-go/internal/config"
	"sentinel-ai/ingestor-go/internal/kafka"
	"sentinel-ai/ingestor-go/internal/metrics"
	"sentinel-ai/ingestor-go/internal/server"
)

func main() {
	cfg := config.FromEnv()

	met := metrics.New()

	producer, err := kafka.NewProducer(cfg.KafkaBrokers, cfg.RawFramesTopic, met)
	if err != nil {
		log.Fatalf("Kafka Producer init failed: %v", err)
	}
	defer producer.Close()

	grpcLis, err := net.Listen("tcp", cfg.GRPCAddr)
	if err != nil {
		log.Fatalf("listen failed: %v", err)
	}

	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(32 * 1024 * 1024),
	)
	server.Register(grpcServer, producer, met)

	metricsSrv := &http.Server{
		Addr: cfg.MetricsAddr,
		Handler: met.Handler(),
	}

	// start metrics
	go func() {
		log.Printf("metrics listening on %s", cfg.MetricsAddr)
		if err := metricsSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("metrics server failed: %v", err)
		}
	}()

	// Start gRPC
	go func() {
		log.Printf("gRPC listening on %s", cfg.GRPCAddr)
		if err := grpcServer.Serve(grpcLis); err != nil {
			log.Fatalf("grpc serve failed: %v", err)
		}
	}()

	// Shutdown
	ch := make(chan os.Signal, 2)
	signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
	<-ch

	log.Println("shutting down...")
	grpcServer.GracefulStop()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = metricsSrv.Shutdown(ctx)
}