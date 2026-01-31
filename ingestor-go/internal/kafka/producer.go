package kafka

import (
	"context"
	"time"

	"github.com/segmentio/kafka-go"

	"sentinel-ai/ingestor-go/internal/metrics"
)

type Producer struct {
	w     *kafka.Writer
	topic string
	met   *metrics.Metrics
}

func NewProducer(brokers []string, topic string, met *metrics.Metrics) (*Producer, error) {
	w := &kafka.Writer{
		Addr:         kafka.TCP(brokers...),
		Topic:        topic,
		Balancer:     &kafka.Hash{},
		BatchSize:    50,
		BatchTimeout: 50 * time.Millisecond,
		RequiredAcks: kafka.RequireOne,
		Async:        false,
	}
	return &Producer{w: w, topic: topic, met: met}, nil
}

func (p *Producer) Close() error {
	return p.w.Close()
}

func (p *Producer) ProduceRawFrame(
	ctx context.Context,
	key []byte,
	value []byte,
	headers map[string]string,
) error {
	kHeaders := make([]kafka.Header, 0, len(headers))
	for k, v := range headers {
		kHeaders = append(kHeaders, kafka.Header{Key: k, Value: []byte(v)})
	}

	msg := kafka.Message{
		Key:     key,
		Value:   value,
		Headers: kHeaders,
		Time:    time.Now(),
	}

	err := p.w.WriteMessages(ctx, msg)
	if err != nil {
		p.met.KafkaErrors.Inc()
	}
	return err
}
