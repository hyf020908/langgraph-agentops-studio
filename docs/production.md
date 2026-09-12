# Backend operations and deployment

The backend supports two execution modes. `inline` retains the original CLI/API
behavior with bounded admission. `rabbitmq` persists a job and outbox record in
one SQLite transaction; an independent worker publishes and consumes persistent
messages. Neither mode changes HTML, CSS, or component markup.

## Deployment scope

This release is a **single-host, shared-workspace service**, with multiple worker
processes supported on the same local Docker volume. SQLite checkpoints, job
metadata, BM25 index, and artifacts all require that volume. Do not place SQLite
WAL databases on NFS, or run workers with independent copies of `/data`.
The provided one-node RabbitMQ container is not a highly available broker cluster.
Quorum queues can use a separately operated three-node RabbitMQ cluster.

Authentication, per-user ownership, tenant isolation, distributed PostgreSQL
checkpoint/job storage, and object storage are not implemented here. Do not expose
this shared-workspace API directly to the public internet. Place it behind an
authenticated gateway with TLS, request-size limits, access policy and rate limits.
Task IDs are identifiers, not authorization. `/ingest`, `/metrics`, `/providers`
and approval operations require appropriate gateway access rules.

## Start

1. Create `.env` from `.env.example`; configure LLM/embedding credentials and a
   strong RabbitMQ password (URL-encode reserved characters in AMQP credentials).
2. Run `docker compose up --build -d`. API listens on `127.0.0.1:8000`; RabbitMQ
   management listens on `127.0.0.1:15672`. Qdrant and AMQP are internal.
3. Run `docker compose exec api python -m app.ingest --source-dir /knowledge`.
4. Check `/health`, `/ready`, worker logs and RabbitMQ consumer count. Readiness
   verifies runtime/checkpoint access; it does not make paid provider calls or
   certify that a consumer is online.
5. Increase worker replicas cautiously using `docker compose up -d --scale worker=2`.
   This scales on one host only. Each worker consumes at most `MAX_CONCURRENT_RUNS`
   deliveries. Total provider pressure grows with replicas.

Use `pip install -r requirements.lock` then `pip install --no-deps -e .` for the
tested Python 3.13 dependency snapshot. The lock is a version snapshot, not a
hash-verified supply-chain lock. Container image tags should be digest-pinned in
the deployment environment after platform-specific validation.

## Request and delivery contracts

The browser uses `POST /runs?wait=false` and `POST /runs/{id}/continue?wait=false`.
Queue acceptance returns HTTP 202 with `status=queued`. Poll `GET /runs/{id}`
until `queue_status=done|failed`. `done` means the submitted operation finished;
the workflow may now need human approval. The page already contains that gate.
Default `wait=true` remains compatible with old clients using async waiting up
to 15 minutes. A wait timeout/disconnect does not cancel the durable job.

- Invalid request models return 422; unknown runs 404; duplicate IDs/active
  continuation 409; overload 503 with `Retry-After: 2`; unexpected failures 500
  with a generic public message and `X-Request-ID`.
- A repeated start with the same ID is rejected, not silently executed again.
  After a lost HTTP response, inspect the known task ID before resubmitting.
- Outbox publication uses mandatory routing, persistent messages and publisher
  confirms. Unconfirmed publications remain retryable. Confirmed-but-unconsumed
  jobs can be republished after 60 seconds; consumer deduplication is required.
- Manual ACK follows persisted job completion. AMQP heartbeats run on the I/O
  thread while graph calls run in a bounded executor. Prefetch equals concurrency.
- Per-task OS locks protect duplicate execution across processes on the local
  volume. A process crash releases its lock; RabbitMQ redelivers its unacked job.
  Workers inspect existing checkpoints before replay and cap recovery attempts.
- Continuations carry the checkpoint ID that was reviewed. A redelivery cannot
  apply that same approval to a different later human gate.
- Execution errors are recorded as failed jobs and negatively acknowledged to
  `agentops.runs.dead`. They are not blindly retried as fresh workflows. Inspect
  logs/checkpoints and resolve the cause before an operator starts replacement
  work. No automatic side-effect-safe replay claim is made: an external call can
  succeed before its checkpoint commits.

RabbitMQ has a delivery limit of 5. Broker-level exhaustion can dead-letter a job
before the application marks it failed; alert on dead-letter depth and investigate
stale `running` records. Never purge or replay a dead-letter queue automatically.
If all workers are down, accepted jobs remain in the outbox and consume the
bounded backlog (`MAX_PENDING_JOBS`, default 1000). Alert on oldest queued age.

## Capacity and failure controls

Inline HTTP admission reserves capacity for read requests; queue mode allows 96
concurrent POST requests and 128 total requests per API process. The normal browser
does not hold POST connections open in queue mode. Health is handled asynchronously
and bypasses admission. Model and embedding providers each allow 8 concurrent
calls per process, use two SDK retries, and open a circuit after 5 transient
failures for 30 seconds. One recovery probe is allowed. SDK request timeout is
not an overall workflow deadline; size graceful shutdown and broker consumer
acknowledgement timeouts for measured workflow durations.

Artifacts use same-directory temporary files, fsync and atomic replace. This
prevents partial JSON reads; it is not a multi-file transaction or a guarantee
against all filesystem/power-loss failures. Back up the full data volume with
workers and ingestion stopped, or use SQLite-aware online backups. Test restore.
SQLite failure now fails explicitly rather than silently losing persistence by
falling back to in-memory checkpoints.

Monitor `/metrics`, HTTP error/overload rates, worker errors, RabbitMQ ready and
unacked depth, dead-letter depth, oldest job age, provider latency/429s, disk space,
and checkpoint size. Request metrics are per-process totals. Provision a retention
policy for completed jobs, checkpoints, artifacts and locks; no destructive automatic
cleanup is enabled. Do not infer throughput from configured thread count: use
representative tasks and provider quotas to measure queue latency and completion
p50/p95/p99 before increasing concurrency.

## Hybrid retrieval

Ingestion writes a persistent SQLite FTS5 BM25 index using the same point IDs as
the vector store. Latin technical tokens and CJK characters/bigrams are indexed
and queried identically. Chinese tokenization is lightweight, not linguistic
word segmentation. Upserts replace identical IDs; old chunks from deleted or
shortened documents require an offline rebuild of both indexes.

Search obtains up to `RAG_CANDIDATE_K=30` candidates per lane, merges by stable
point ID and computes `RRF(d) = sum(1 / (RAG_RRF_K + rank_lane(d)))`, default k=60.
Each lane contributes once per document. Raw vector and BM25 scores are never
directly added. The fused candidate pool is bounded before second-stage reranking.
`RAG_TOP_K` limits the returned evidence. `RAG_SCORE_THRESHOLD` applies only to
vector recall, not to RRF scores.

`RAG_RERANK_PROVIDER=http` invokes a learned rerank endpoint (for example a hosted
BGE reranker) configured with `RAG_RERANK_URL`, `RAG_RERANK_MODEL` and optional
`RAG_RERANK_API_KEY`. Request contract:

```json
{"model":"BAAI/bge-reranker-v2-m3","query":"question","documents":["chunk"],"top_n":4}
```

Response contract: `{"results":[{"index":0,"relevance_score":0.95}]}`. Indexes,
cardinality and finite scores are validated. Timeout/error/malformed output falls
back to RRF, not a failed research run. The endpoint is deployment configuration,
never a user-controlled URL. Retrieved text is sent to this endpoint, so select
a provider authorized for your knowledge corpus.

Default `lexical` reranking uses token coverage plus RRF. It is a deterministic
fallback, **not a learned semantic cross-encoder**. `none` disables the second
stage. Configure `http` to use model-based reranking; no model weights or remote
rerank service are downloaded implicitly. Metadata preserves lane ranks, RRF score
and rerank score. If embeddings fail, BM25 still works; if BM25 fails, vectors
still work. Both lanes failing returns no local evidence and logs degradation.

Embedding ingestion/upsert runs in batches of 32 with response shape validation.
Documents are limited to 2 MiB each and 1000 per operation. BM25 is committed before
embedding so it remains available on embedding failure; the two indexes are not
one transaction. Retry ingestion after resolving failure. Recreate collections
only during offline maintenance; HTTP recreation is rejected. HTTP source paths
must stay inside `RAG_SOURCE_DIR`; escaping symlinks are excluded.

## Prompts and validation

Shared prompt contract version 2.0.0 defines evidence/instruction boundaries,
uncertainty, non-fabrication, language and JSON requirements. Planner, analyst and
reviewer have role-specific acceptance and risk rules. The reviewer receives
actual evidence and a truncation flag, rather than only evidence count. Unknown
finding evidence IDs are rejected by code. Human-required approval cannot be
waived by the review model. Research/executor templates are maintained for future
model-backed variants; current research tools and artifact rendering remain
deterministic and do not call those two templates.

## Validation

Run `python -m pytest -q -p no:cacheprovider` and `cd web && npm ci && npm run build`.
Tests cover concurrent admission, transactional enqueue, publication uncertainty,
worker dedup/recovery, SQLite reopen, atomic artifacts, BM25 persistence, RRF,
rerank response validation/fallback, queue API compatibility and health under
overload. Real provider tests require explicit credentials/opt-in.

For an isolated live broker, set `RABBITMQ_TEST_URL` and run
`python -m pytest tests/integration/test_rabbitmq_delivery.py -q`. This checks
publisher confirms, durable message properties, redelivery after channel loss,
and ACK using uniquely named queues that are deleted in cleanup. It is skipped
unless explicitly configured; mocked delivery tests do not replace this check.

RabbitMQ protocol references: [publisher confirms and acknowledgements](https://www.rabbitmq.com/docs/confirms),
[reliability and duplicate delivery](https://www.rabbitmq.com/docs/reliability),
[Pika heartbeat/threading constraints](https://pika.readthedocs.io/en/stable/modules/adapters/blocking.html).
SQLite saver construction follows the [LangGraph reference](https://reference.langchain.com/python/langgraph.checkpoint.sqlite/SqliteSaver).
