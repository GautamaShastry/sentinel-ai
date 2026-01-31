import os
import time
import uuid
import cv2
import grpc

def _imports():
    from video_pb2 import FrameRequest
    from video_pb2_grpc import VideoStreamerStub
    return FrameRequest, VideoStreamerStub

def frame_generator(camera_id: str, device_index: int, fps: int, jpeg_quality: int):
    FrameRequest, _ = _imports()
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError("could not open webcam")

    seq = 0
    delay = 1.0 / max(1, fps)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        ts_ms = int(time.time() * 1000)
        frame_id = str(uuid.uuid4())

        ok2, jpg = cv2.imencode(".jpg", frame, encode_params)
        if not ok2:
            continue

        yield FrameRequest(
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp_ms=ts_ms,
            sequence_number=seq,
            encoding="jpeg",
            image_data=jpg.tobytes(),
        )

        seq += 1
        time.sleep(delay)


def main():
    grpc_target = os.getenv("GRPC_TARGET", "localhost:50051")
    camera_id = os.getenv("CAMERA_ID", "camera_1")
    device_index = int(os.getenv("DEVICE_INDEX", "0"))
    fps = int(os.getenv("FPS", "25"))
    jpeg_quality = int(os.getenv("JPEG_QUALITY", "70"))

    FrameRequest, VideoStreamerStub = _imports()

    channel = grpc.insecure_channel(grpc_target, options=[
        ("grpc.max_send_message_length", 32 * 1024 * 1024),
        ("grpc.max_receive_message_length", 32 * 1024 * 1024),
    ])
    stub = VideoStreamerStub(channel)

    print(f"[client] streaming to {grpc_target} as {camera_id} at {fps} fps")
    status = stub.UploadVideo(frame_generator(camera_id, device_index, fps, jpeg_quality))
    print("[client] server status:", status)

if __name__ == "__main__":
    main()
