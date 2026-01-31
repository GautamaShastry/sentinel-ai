import asyncio
import json
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from app.db import fetch_all, fetch_one
from app.consumer import LATEST, LATEST_ALERTS

router = APIRouter()

@router.get("/health")
def health():
    return {"ok": True}

@router.get("/cameras")
def cameras():
    rows = fetch_all("SELECT DISTINCT camera_id FROM detections ORDER BY camera_id ASC")
    db_cams = [r["camera_id"] for r in rows]
    mem_cams = list(LATEST.keys())
    cams = sorted(set(db_cams + mem_cams))
    return {"cameras": cams}

@router.get("/detections")
def detections(
    camera_id: str = Query(...),
    since_ms: int = Query(0),
    limit: int = Query(50, ge=1, le=500),
):
    rows = fetch_all(
        """
        SELECT camera_id, frame_id, timestamp_ms, sequence_number, objects, created_at
        FROM detections
        WHERE camera_id = %s AND timestamp_ms >= %s
        ORDER BY timestamp_ms DESC
        LIMIT %s
        """,
        (camera_id, since_ms, limit),
    )
    return {"items": rows}

@router.get("/latest")
def latest(camera_id: str = Query(...)):
    return {"item": LATEST.get(camera_id)}

@router.get("/analytics/summary")
def analytics_summary(camera_id: str = Query(...)):
    """Get detection summary stats for a camera."""
    # Total frames processed
    total = fetch_one(
        "SELECT COUNT(*) as count FROM detections WHERE camera_id = %s",
        (camera_id,)
    )
    
    # Object counts (last hour)
    one_hour_ago = fetch_one("SELECT EXTRACT(EPOCH FROM NOW() - INTERVAL '1 hour') * 1000 as ts")
    since_ts = int(one_hour_ago["ts"]) if one_hour_ago else 0
    
    rows = fetch_all(
        """
        SELECT objects FROM detections 
        WHERE camera_id = %s AND timestamp_ms >= %s
        """,
        (camera_id, since_ts)
    )
    
    object_counts = {}
    for row in rows:
        objs = row.get("objects") or []
        for obj in objs:
            label = obj.get("label", "unknown")
            object_counts[label] = object_counts.get(label, 0) + 1
    
    return {
        "camera_id": camera_id,
        "total_frames": total["count"] if total else 0,
        "object_counts_last_hour": object_counts,
        "unique_objects": list(object_counts.keys()),
    }

@router.get("/analytics/timeline")
def analytics_timeline(
    camera_id: str = Query(...),
    minutes: int = Query(60, ge=1, le=1440),
    bucket_minutes: int = Query(5, ge=1, le=60),
):
    """Get object detection counts over time buckets."""
    rows = fetch_all(
        f"""
        SELECT 
            DATE_TRUNC('minute', to_timestamp(timestamp_ms/1000)) 
                - (EXTRACT(MINUTE FROM to_timestamp(timestamp_ms/1000))::int % %s) * INTERVAL '1 minute' as bucket,
            objects
        FROM detections 
        WHERE camera_id = %s 
          AND timestamp_ms >= EXTRACT(EPOCH FROM NOW() - INTERVAL '{minutes} minutes') * 1000
        ORDER BY bucket
        """,
        (bucket_minutes, camera_id)
    )
    
    buckets = {}
    for row in rows:
        bucket = str(row["bucket"])
        if bucket not in buckets:
            buckets[bucket] = {"timestamp": bucket, "total": 0, "objects": {}}
        
        objs = row.get("objects") or []
        buckets[bucket]["total"] += len(objs)
        for obj in objs:
            label = obj.get("label", "unknown")
            buckets[bucket]["objects"][label] = buckets[bucket]["objects"].get(label, 0) + 1
    
    return {"timeline": list(buckets.values())}


@router.get("/stream/detections")
async def stream_detections(camera_id: str = Query(...), poll_ms: int = Query(250, ge=50, le=2000)):
    async def event_gen():
        last_frame_id = None
        while True:
            item = LATEST.get(camera_id)
            if item and item.get("frame_id") != last_frame_id:
                last_frame_id = item.get("frame_id")
                yield f"data: {json.dumps(item)}\n\n"
            await asyncio.sleep(poll_ms / 1000.0)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/alerts")
def get_alerts(
    camera_id: str = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """Get recent alerts."""
    if camera_id:
        rows = fetch_all(
            """
            SELECT id, camera_id, frame_id, timestamp_ms, alert_type, objects, clip_path, acknowledged, created_at
            FROM alerts
            WHERE camera_id = %s
            ORDER BY timestamp_ms DESC
            LIMIT %s
            """,
            (camera_id, limit),
        )
    else:
        rows = fetch_all(
            """
            SELECT id, camera_id, frame_id, timestamp_ms, alert_type, objects, clip_path, acknowledged, created_at
            FROM alerts
            ORDER BY timestamp_ms DESC
            LIMIT %s
            """,
            (limit,),
        )
    return {"alerts": rows}

@router.get("/alerts/recent")
def get_recent_alerts():
    """Get recent alerts from memory (fast)."""
    return {"alerts": LATEST_ALERTS[:20]}

@router.get("/analytics/electronics")
def analytics_electronics(
    camera_id: str = Query(None),
    hours: int = Query(24, ge=1, le=168),
):
    """Get electronics detection stats."""
    if camera_id:
        rows = fetch_all(
            f"""
            SELECT objects, timestamp_ms
            FROM detections 
            WHERE camera_id = %s 
              AND has_electronics = true
              AND timestamp_ms >= EXTRACT(EPOCH FROM NOW() - INTERVAL '{hours} hours') * 1000
            ORDER BY timestamp_ms DESC
            """,
            (camera_id,)
        )
    else:
        rows = fetch_all(
            f"""
            SELECT camera_id, objects, timestamp_ms
            FROM detections 
            WHERE has_electronics = true
              AND timestamp_ms >= EXTRACT(EPOCH FROM NOW() - INTERVAL '{hours} hours') * 1000
            ORDER BY timestamp_ms DESC
            """
        )
    
    electronics_counts = {}
    total_detections = 0
    for row in rows:
        objs = row.get("objects") or []
        for obj in objs:
            if obj.get("is_electronic"):
                label = obj.get("label", "unknown")
                electronics_counts[label] = electronics_counts.get(label, 0) + 1
                total_detections += 1
    
    return {
        "total_electronics_detections": total_detections,
        "by_type": electronics_counts,
        "hours": hours,
    }

@router.get("/stream/alerts")
async def stream_alerts(poll_ms: int = Query(500, ge=100, le=5000)):
    """SSE stream for real-time alerts."""
    async def event_gen():
        last_count = 0
        while True:
            current_count = len(LATEST_ALERTS)
            if current_count > last_count:
                new_alerts = LATEST_ALERTS[:current_count - last_count]
                for alert in reversed(new_alerts):
                    yield f"data: {json.dumps(alert)}\n\n"
                last_count = current_count
            await asyncio.sleep(poll_ms / 1000.0)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
