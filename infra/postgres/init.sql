CREATE TABLE IF NOT EXISTS detections (
    id BIGSERIAL PRIMARY KEY,
    camera_id TEXT NOT NULL,
    frame_id TEXT NOT NULL,
    timestamp_ms BIGINT NOT NULL,
    sequence_number BIGINT NOT NULL,
    objects JSONB NOT NULL,
    has_electronics BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT Now(),
    UNIQUE(camera_id, frame_id)
);

CREATE INDEX IF NOT EXISTS idx_detections_camera_ts
ON detections (camera_id, timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    camera_id TEXT NOT NULL,
    frame_id TEXT NOT NULL,
    timestamp_ms BIGINT NOT NULL,
    alert_type TEXT NOT NULL,
    objects JSONB NOT NULL,
    clip_path TEXT,
    acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT Now()
);

CREATE INDEX IF NOT EXISTS idx_alerts_camera_ts
ON alerts (camera_id, timestamp_ms DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_type
ON alerts (alert_type, created_at DESC);