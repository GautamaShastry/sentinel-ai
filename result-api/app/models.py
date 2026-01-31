from pydantic import BaseModel
from typing import Any

class Detection(BaseModel):
    camera_id: str
    frame_id: str
    timestamp_ms: int
    sequence_number: int
    objects: list[dict[str, Any]]
