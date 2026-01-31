package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type Metrics struct {
	reg *prometheus.Registry

	FramesIngested prometheus.Counter
	BytesIngested  prometheus.Counter
	KafkaErrors    prometheus.Counter
}

func New() *Metrics {
	reg := prometheus.NewRegistry()

	m := &Metrics{
		reg: reg,
		FramesIngested: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "frames_ingested_total",
			Help: "Total frames ingested via gRPC",
		}),
		BytesIngested: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "ingested_bytes_total",
			Help: "Total bytes ingested via gRPC",
		}),
		KafkaErrors: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "kafka_errors_total",
			Help: "Total kafka produce errors",
		}),
	}

	reg.MustRegister(m.FramesIngested, m.BytesIngested, m.KafkaErrors)
	return m
}

func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.reg, promhttp.HandlerOpts{})
}
