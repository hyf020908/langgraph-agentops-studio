"""Run with python -m app.worker. AMQP I/O stays separate from long graph calls."""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import signal
from threading import Event, Thread

import portalocker

from app.runner import WorkflowRunner
from schemas.models import RunResponse
from services.broker import connect, topology, publish_pending
from services.config import load_settings
from services.jobs import JobStore

logger = logging.getLogger("agentops.worker")


def execute_job(store, runner, identifier):
    job = store.get(identifier)
    if job is None:
        return False  # Unknown/poison messages go to the dead-letter queue.
    lock_dir = store.root / "locks"
    lock_dir.mkdir(exist_ok=True)
    try:
        with portalocker.Lock(str(lock_dir / (job["task_id"] + ".lock")), timeout=0):
            job = store.get(identifier)
            if job["status"] in {"done", "failed"}:
                return job["status"] == "done"
            store.running(identifier)
            try:
                if job["attempts"] >= 3:
                    raise RuntimeError("Worker recovery attempt limit reached.")
                payload = json.loads(job["payload"])
                config = {"configurable": {"thread_id": job["task_id"]}}
                snapshot = runner.graph.get_state(config)
                if not snapshot.values:
                    if job["operation"] != "start":
                        raise RuntimeError("Cannot resume a missing checkpoint.")
                    state, interrupt = runner.start(**payload)
                elif job["operation"] == "continue" and snapshot.config["configurable"].get("checkpoint_id") == payload.pop("checkpoint_id", None):
                    state, interrupt = runner.continue_run(**payload)
                else:
                    state, interrupt = runner.inspect(job["task_id"])
                    if snapshot.next and interrupt is None:
                        state = runner.graph.invoke(None, config=config)
                        interrupt = runner._extract_interrupt_payload(state)
                result = runner.summarize(state, interrupt).model_dump(mode="json")
                store.finish(identifier, result, failed=result["status"] == "failed")
                return result["status"] != "failed"
            except Exception:
                logger.exception("job_id=%s task_id=%s execution failed", identifier, job["task_id"])
                result = RunResponse(task_id=job["task_id"], status="failed", approval_required=False,
                                     review_summary="Execution failed; operator investigation required.")
                store.finish(identifier, result.model_dump(mode="json"), failed=True)
                return False
    except portalocker.exceptions.LockException:
        # A live owner holds the original unacked delivery. ACK this duplicate;
        # the owner's delivery is redelivered if that process dies.
        return True


def dispatch(store, stop):
    while not stop.is_set():
        connection = None
        try:
            connection = connect()
            channel = connection.channel()
            queue = topology(channel)
            channel.confirm_delivery()
            while not stop.is_set():
                publish_pending(store, channel, queue)
                connection.process_data_events(time_limit=1)
                stop.wait(1)
        except Exception:
            logger.exception("Outbox publisher disconnected; retrying in five seconds.")
            stop.wait(5)
        finally:
            if connection is not None and connection.is_open:
                connection.close()


def main():
    settings = load_settings()
    if settings.checkpoint_mode != "sqlite":
        raise ValueError("RabbitMQ workers require CHECKPOINT_MODE=sqlite.")
    if settings.vector_db.provider == "qdrant" and not settings.vector_db.qdrant_url:
        raise ValueError("RabbitMQ workers require server Qdrant; set QDRANT_URL.")
    runner = WorkflowRunner()
    store = JobStore(settings.output_root)
    stop = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    publisher = Thread(target=dispatch, args=(store, stop), name="outbox-publisher")
    publisher.start()
    concurrency = settings.max_concurrent_runs
    try:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="workflow") as pool:
            while not stop.is_set():
                connection = None
                futures = set()
                try:
                    connection = connect()
                    channel = connection.channel()
                    queue = topology(channel)
                    channel.basic_qos(prefetch_count=concurrency)

                    def received(ch, method, properties, body):
                        try:
                            identifier = body.decode("ascii")
                            if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
                                raise ValueError("Invalid job ID")
                        except (UnicodeError, ValueError):
                            ch.basic_nack(method.delivery_tag, requeue=False)
                            return
                        future = pool.submit(execute_job, store, runner, identifier)
                        futures.add(future)

                        def completed(done, conn=connection, channel=ch, tag=method.delivery_tag):
                            def acknowledge():
                                futures.discard(done)
                                if not channel.is_open:
                                    return
                                try:
                                    successful = done.result()
                                except Exception:
                                    logger.exception("Worker storage failure; delivery will be requeued.")
                                    channel.basic_nack(tag, requeue=True)
                                    return
                                if successful:
                                    channel.basic_ack(tag)
                                else:
                                    channel.basic_nack(tag, requeue=False)
                            try:
                                if conn.is_open:
                                    conn.add_callback_threadsafe(acknowledge)
                            except Exception:
                                logger.warning("ACK could not be scheduled; broker will redeliver.")
                        future.add_done_callback(completed)

                    tag = channel.basic_consume(queue, received, auto_ack=False)
                    while not stop.is_set():
                        connection.process_data_events(time_limit=1)
                    channel.basic_cancel(tag)
                    # Keep heartbeats and ACK callbacks flowing during drain.
                    while futures:
                        connection.process_data_events(time_limit=1)
                except Exception:
                    logger.exception("Consumer disconnected; draining active work before reconnect.")
                    for future in list(futures):
                        try:
                            future.result()
                        except Exception:
                            logger.exception("Active task failed while disconnected.")
                    stop.wait(5)
                finally:
                    if connection is not None and connection.is_open:
                        connection.close()
    finally:
        stop.set()
        publisher.join()
        runner.close()


if __name__ == "__main__":
    main()
