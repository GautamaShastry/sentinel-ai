import json
from confluent_kafka import Consumer, Producer

def build_consumer(bootstrap: str, group_id: str):
    return Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": group_id,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "max.poll.interval.ms": 600000,
    })

def build_producer(bootstrap: str):
    return Producer({
        "bootstrap.servers": bootstrap,
        "linger.ms": 10,
        "batch.num.messages": 1000,
    })

def headers_to_dict(headers):
    d = {}
    if not headers:
        return d
    for k, v in headers:
        if isinstance(v, (bytes, bytearray)):
            d[k] = v.decode("utf-8", errors="ignore")
        else:
            d[k] = str(v)
    return d

from typing import Optional

def produce_json(producer: Producer, topic: str, key: str, payload: dict, headers: Optional[dict] = None):
    hdrs = []
    if headers:
        hdrs = [(k, str(v).encode("utf-8")) for k, v in headers.items()]

    producer.produce(
        topic=topic,
        key=key.encode("utf-8"),
        value=json.dumps(payload).encode("utf-8"),
        headers=hdrs,
    )
