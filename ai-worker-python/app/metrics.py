from prometheus_client import Counter, Histogram, start_http_server

FRAMES_CONSUMED = Counter("frames_consumed_total", "Total frames consumed from Kafka")
FRAMES_PROCESSED = Counter("frames_processed_total", "Total frames processed successfully")
FRAMES_DLQ = Counter("frames_dlq_total", "Total frames sent to DLQ")
INFERENCE_SECONDS = Histogram("inference_seconds", "YOLO inference time in seconds")

def start_metrics_server(port: int):
    start_http_server(port)
