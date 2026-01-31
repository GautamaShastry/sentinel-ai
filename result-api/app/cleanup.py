import os
import time
import threading
from app.db import execute

# Retention settings
DETECTION_RETENTION_HOURS = int(os.getenv("DETECTION_RETENTION_HOURS", "24"))
ALERT_RETENTION_HOURS = int(os.getenv("ALERT_RETENTION_HOURS", "168"))  # 7 days
CLEANUP_INTERVAL_SEC = int(os.getenv("CLEANUP_INTERVAL_SEC", "3600"))  # 1 hour

def cleanup_old_records():
    """Delete old detections and alerts."""
    try:
        # Delete old detections
        result = execute(
            f"""
            DELETE FROM detections 
            WHERE created_at < NOW() - INTERVAL '{DETECTION_RETENTION_HOURS} hours'
            """
        )
        
        # Delete old alerts (keep longer)
        result = execute(
            f"""
            DELETE FROM alerts 
            WHERE created_at < NOW() - INTERVAL '{ALERT_RETENTION_HOURS} hours'
            """
        )
        
        print(f"[cleanup] Cleaned up records older than {DETECTION_RETENTION_HOURS}h (detections) / {ALERT_RETENTION_HOURS}h (alerts)")
    except Exception as e:
        print(f"[cleanup] Error: {e}")

def start_cleanup_loop():
    """Run cleanup periodically."""
    def loop():
        while True:
            time.sleep(CLEANUP_INTERVAL_SEC)
            cleanup_old_records()
    
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f"[cleanup] Started cleanup job (every {CLEANUP_INTERVAL_SEC}s)")
