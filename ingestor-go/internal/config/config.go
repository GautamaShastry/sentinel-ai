package config

import (
	"os"
	"strings"
)

type Config struct {
	GRPCAddr       string
	MetricsAddr    string
	KafkaBrokers   []string
	RawFramesTopic string
}

func FromEnv() Config {
	grpcAddr := getenv("GRPC_ADDR", "0.0.0.0:50051")
	metricsAddr := getenv("METRICS_ADDR", "0.0.0.0:9100")
	brokers := strings.Split(getenv("KAFKA_BROKERS", "localhost:19092"), ",")
	topic := getenv("RAW_FRAMES_TOPIC", "raw-frames")

	return Config{
		GRPCAddr:       grpcAddr,
		MetricsAddr:    metricsAddr,
		KafkaBrokers:   brokers,
		RawFramesTopic: topic,
	}
}

func getenv(k, def string) string {
	v := os.Getenv(k)
	if v == "" {
		return def
	}
	return v
}
