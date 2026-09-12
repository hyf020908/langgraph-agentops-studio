"""RabbitMQ topology and publisher-confirm protocol."""
import os
import pika


def queue_enabled() -> bool:
    mode = os.getenv("TASK_QUEUE_MODE", "inline")
    if mode not in {"inline", "rabbitmq"}:
        raise ValueError("TASK_QUEUE_MODE must be inline or rabbitmq.")
    return mode == "rabbitmq"


def connect():
    params = pika.URLParameters(os.environ["RABBITMQ_URL"])
    params.heartbeat = 30
    params.blocked_connection_timeout = 10
    params.socket_timeout = 10
    params.stack_timeout = 15
    params.connection_attempts = 1
    return pika.BlockingConnection(params)


def topology(channel) -> str:
    queue = os.getenv("RABBITMQ_QUEUE", "agentops.runs")
    channel.queue_declare(queue=queue + ".dead", durable=True, arguments={"x-queue-type": "quorum"})
    channel.queue_declare(queue=queue, durable=True, arguments={
        "x-queue-type": "quorum", "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": queue + ".dead",
        "x-overflow": "reject-publish", "x-max-length": 10000,
        "x-delivery-limit": 5,
    })
    return queue


def publish_pending(store, channel, queue):
    for job in store.pending():
        channel.basic_publish(exchange="", routing_key=queue, body=job["id"].encode(), mandatory=True,
                              properties=pika.BasicProperties(delivery_mode=2, content_type="text/plain", message_id=job["id"]))
        store.published(job["id"])
