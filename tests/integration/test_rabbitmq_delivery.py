"""Explicit opt-in broker test; creates and removes only uniquely named test queues."""
import os
import time
from uuid import uuid4

import pytest

from services.broker import connect, topology, publish_pending
from services.jobs import JobStore


@pytest.mark.skipif(not os.getenv("RABBITMQ_TEST_URL"), reason="Set RABBITMQ_TEST_URL to an isolated test broker.")
def test_persistent_confirmed_delivery_is_redelivered_until_ack(tmp_path, monkeypatch):
    monkeypatch.setenv("RABBITMQ_URL", os.environ["RABBITMQ_TEST_URL"])
    queue = "agentops-test-" + uuid4().hex
    monkeypatch.setenv("RABBITMQ_QUEUE", queue)
    store = JobStore(str(tmp_path))
    job = store.enqueue("delivery-test", "start", {"task": "test"})
    connection = connect()
    try:
        channel = connection.channel()
        topology(channel)
        channel.confirm_delivery()
        publish_pending(store, channel, queue)
        assert store.get(job["id"])["published_at"] is not None
        method, properties, body = channel.basic_get(queue, auto_ack=False)
        assert body.decode() == job["id"]
        assert properties.delivery_mode == 2
        channel.close()  # Simulate a consumer disappearing before ACK.
        channel = connection.channel()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            method, properties, body = channel.basic_get(queue, auto_ack=False)
            if method is not None:
                break
            connection.sleep(.1)
        assert method is not None and method.redelivered
        assert body.decode() == job["id"]
        channel.basic_ack(method.delivery_tag)
        connection.process_data_events(time_limit=.1)
        assert channel.basic_get(queue, auto_ack=True)[0] is None
    finally:
        if connection.is_open:
            cleanup = connection.channel()
            cleanup.queue_delete(queue)
            cleanup.queue_delete(queue + ".dead")
            connection.close()
