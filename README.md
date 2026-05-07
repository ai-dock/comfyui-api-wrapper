# ComfyUI API Wrapper

A FastAPI service that fronts a ComfyUI instance (or several) with
a job-queue API. Customers submit a workflow via `POST /generate`
or one of its variants; the wrapper preprocesses it (downloading
any URL-shaped inputs), submits it to ComfyUI, awaits completion
over the WebSocket, postprocesses the output files (S3 upload,
optional webhook), and returns a structured result envelope.

**Source code:** [github.com/ai-dock/comfyui-api-wrapper](https://github.com/ai-dock/comfyui-api-wrapper)

## What it does

- Three submit shapes: async (`POST /generate` returns 202 + a
  request id), sync (`POST /generate/sync` blocks until done),
  and SSE-streaming (`POST /generate/stream` emits progress
  events). Same payload, same Result shape, different delivery.
- Two ways to specify a workflow: raw ComfyUI API-format JSON
  in `input.workflow_json`, or a curated **modifier** class
  + `modifications` dict that the modifier applies to a baked-in
  workflow.
- Automatic URL-fetch on input nodes: any string in the workflow
  that parses as a URL is downloaded (MD5-hashed cache,
  MIME-type-driven extension) into ComfyUI's `input/` directory
  and replaced with the local filename.
- Optional output delivery in three independent modes (combine
  freely): S3 upload (presigned `url` per output), inline base64
  (`data` per output, opt-in), and fire-and-forget webhook
  delivery of a slim summary envelope (HMAC-SHA256 signed when
  a secret is configured — see [§ Webhook](#webhook)).
- Three-stage internal pipeline (preprocess → generation →
  postprocess) with separate worker pools; bounded inbound queue
  with HTTP 503 + `Retry-After` when at capacity.
- Multi-backend pool: drive N ComfyUI processes from one wrapper
  for parallelism. See [§ Multi-backend](#multi-backend).
- WebSocket reconnect, ComfyUI-side cache-validation,
  OOM-recovery via `/free`, CUDA-fault → 503 health latch — see
  [§ Reliability](#reliability) for the failure modes the wrapper
  handles for you.

## Quick start

```bash
# 1. Dependencies
apt-get install -y libmagic1                 # for python-magic
pip install -r requirements.txt

# 2. Run ComfyUI somewhere (default: 127.0.0.1:8188)

# 3. Run the wrapper
uvicorn main:app --host 0.0.0.0 --port 8000

# 4. Submit a job
curl -X POST http://localhost:8000/generate/sync \
  -H 'Content-Type: application/json' \
  -d '{"input": {"workflow_json": { ... }}}'
```

`.env.example` ships in the repo as a template for environment
configuration. Most defaults are sensible.

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/generate`              | POST | Submit async — returns 202 + Result with `status: "queued"`. Caller polls `/result/{id}` or supplies a `webhook` for delivery. |
| `/generate/sync`         | POST | Submit and block until terminal (`completed`/`failed`/`timeout`/`cancelled`). Cancel-on-disconnect: closing the connection mid-job marks the request cancelled and the worker queue skips remaining stages. |
| `/generate/stream`       | POST | Submit and stream Server-Sent Events with progress + queue-position updates, ending with a final `final_result` event. |
| `/result/{id}`           | GET  | Read a previously-submitted job's current Result. 404 if unknown. |
| `/cancel/{id}`           | POST | Soft-cancel: flip the Result status to `cancelled`. The wrapper queue checks this marker before processing each stage and skips ahead; a generation already in flight on ComfyUI is interrupted via ComfyUI's `/api/interrupt`. |
| `/queue-info`            | GET  | Current sizes of preprocess / generation / postprocess queues. |
| `/health`                | GET  | Liveness probe (200/503). See [§ Health endpoint](#health-endpoint). |
| `/`                      | GET  | This README rendered as HTML. |
| `/docs`                  | GET  | FastAPI's interactive OpenAPI docs. |

## Request payload

```json5
{
  "input": {
    "request_id": "optional-uuid-v4",     // server-generated if omitted

    // EITHER raw workflow…
    "workflow_json": { /* ComfyUI API-format workflow */ },

    // …OR a curated template (mutually exclusive with workflow_json):
    "modifier": "ModifierClassName",      // e.g. "Image2Image"
    "modifications": { "prompt": "...", "seed": 42 },

    // Optional: per-request S3 override of the env defaults.
    "s3": {
      "access_key_id":     "...",
      "secret_access_key": "...",
      "endpoint_url":      "https://s3.amazonaws.com",
      "bucket_name":       "your-bucket",
      "region":            "us-east-1"
    },

    // Optional: per-request webhook override.
    "webhook": {
      "url":          "https://your-endpoint.example.com",
      "extra_params": {"any": "context"}
    }
  }
}
```

`request_id` is optional; the wrapper mints a UUIDv4 if absent.
The same id flows through the pipeline and into the Result.

`workflow_json` and `modifier` are mutually exclusive — supply
exactly one. `modifications` only applies in modifier mode.

## Result envelope

Returned from `/generate/sync` (inline), `/result/{id}` (after
async completion), and as the `result` payload of the final
`final_result` SSE event:

```json
{
  "id":      "request-uuid",
  "message": "Processing complete.",
  "status":  "queued|processing|generating|generated|completed|failed|timeout|cancelled",
  "comfyui_response": { /* ComfyUI's /history/{prompt_id} payload, slim by default — see INCLUDE_COMFYUI_RESPONSE */ },
  "output":  [
    {
      "filename":    "ComfyUI_00001_.png",
      "local_path":  "/workspace/ComfyUI/output/<request_id>/ComfyUI_00001_.png",
      "subfolder":   "",
      "type":        "output",
      "node_id":     "9",
      "output_type": "images",
      "url":         "https://your-bucket.../ComfyUI_00001_.png",   // only when S3 upload succeeded
      "upload_error": "..."                                          // only on upload failure
    }
  ],
  "timings": {
    "preprocess_ms":  12,
    "generation_ms": 18412,
    "postprocess_ms": 743
  }
}
```

`output[*].url` is populated only when S3 is configured (env or
per-request) and the upload succeeded. With no S3 configured,
files exist only at `local_path` inside the wrapper container.
On upload failure `upload_error` is set and `url` is omitted.

`output[*].data` (base64-encoded bytes) and `output[*].mimetype`
are populated only when the customer opted in via either
`input.return_outputs_as_base64: true` in the body or the
`X-Return-Outputs-As-Base64: 1` request header. Files larger
than `OUTPUT_BASE64_MAX_BYTES` (default 10 MB) are skipped with
`output[*].error` set instead of `data`. Base64 mode coexists
with S3 — both `data` and `url` are populated when both are
configured.

By default `comfyui_response` is `{}` to keep wire weight down.
Set `INCLUDE_COMFYUI_RESPONSE=true` (env) or
`X-Include-ComfyUI-Response: 1` (request header) to receive the
full ComfyUI history blob.

## Streaming (SSE)

Each event is a JSON object on a single `data: …\n\n` line.
Two event shapes — progress and final:

**Progress event** (multiple per job, on status / queue-position
change):

```json
{
  "request_id":   "uuid",
  "status":       "queued|processing|generating|generated",
  "message":      "human-readable status",
  "timestamp":    "2026-05-07T12:34:56.789Z",
  "elapsed_time": 45.2,
  "queue_position": {
    "current_queue":       "preprocessing|generation|postprocessing|processing|unknown",
    "position":            3,
    "queue_size":          5,
    "estimated_wait_time": 360       // seconds; rough heuristic (30s/120s/20s per stage)
  },
  "queue_info": {
    "preprocess_queue_size":  2,
    "generation_queue_size":  1,
    "postprocess_queue_size": 0
  },
  "output_count": 1                  // optional: number of outputs ready so far
}
```

**Final event** (emitted once when the job reaches a terminal
state; this is what your client should look for to close the
stream):

```json
{
  "request_id":   "uuid",
  "status":       "final_result",
  "result":       { /* full Result envelope, see above */ },
  "elapsed_time": 47.1
}
```

`estimated_wait_time` is a rough heuristic — fixed seconds-per-job
multipliers per stage, not a measurement. Treat it as "order of
magnitude" rather than an SLO.

## Examples

### Sync request

```python
import requests

payload = {"input": {"workflow_json": { /* ... */ }}}

r = requests.post("http://localhost:8000/generate/sync", json=payload)
result = r.json()
print(result["status"], result["output"])
```

`/generate/sync` blocks until the job reaches a terminal state.
There is no server-side timeout query parameter — the connection
stays open as long as the worker is making progress.

### Async + poll

```python
r = requests.post("http://localhost:8000/generate", json=payload)
job_id = r.json()["id"]

while True:
    rr = requests.get(f"http://localhost:8000/result/{job_id}")
    data = rr.json()
    if data["status"] in ("completed", "failed", "timeout", "cancelled"):
        break
    time.sleep(2)
```

### Async + webhook

Add `input.webhook.url` to the payload (or set the global
`WEBHOOK_URL` env). After postprocess the wrapper POSTs a slim
summary to that URL — see [§ Webhook](#webhook) for the body
shape and caveats. Consumers needing the full Result (including
`comfyui_response`) should fetch it via `GET /result/{id}` from
their webhook handler.

### Streaming

`POST /generate/stream` returns `text/event-stream`. Use any
SSE client that supports POST + a JSON body — `EventSource`
is GET-only and **will not work** here.

```python
import json, requests, sseclient

r = requests.post("http://localhost:8000/generate/stream", json=payload, stream=True)
client = sseclient.SSEClient(r)
for ev in client.events():
    data = json.loads(ev.data)
    if data["status"] == "final_result":
        print("done:", data["result"])
        break
    pos = data.get("queue_position", {})
    if pos.get("position"):
        print(f"queue {pos['current_queue']} pos {pos['position']}/{pos['queue_size']}")
```

JavaScript: use `fetch()` with `body` and parse the stream
manually, or a library like `@microsoft/fetch-event-source` that
adds POST support to the SSE pattern.

### Cancellation

```python
requests.post(f"http://localhost:8000/cancel/{job_id}")
```

The next stage boundary (or the generation worker's per-5s cancel
check) sees the marker and skips ahead. An in-flight ComfyUI
generation is interrupted via `/api/interrupt`. Cancellation is
soft — the request returns `cancelled` rather than terminating
the connection mid-output.

## Configuration

All configuration is via environment variables (`.env` is loaded
automatically if present in the working directory). Defaults are
in `config/config.py`.

### ComfyUI connection

| Env | Default | Notes |
|---|---|---|
| `COMFYUI_API_BASE`    | `http://127.0.0.1:8188` | Single-backend HTTP base. Overridden by `COMFYUI_BACKENDS` if that's set. |
| `COMFYUI_BACKENDS`    | (unset) | Comma-separated list of backend HTTP bases — see [§ Multi-backend](#multi-backend). When set, the wrapper spawns one generation worker per entry. |
| `COMFYUI_INSTALL_PATH`| `/workspace/ComfyUI`   | Root of the ComfyUI install. Used to locate `input/` and `output/`. |

### WebSocket timeouts

The wrapper waits on ComfyUI's WebSocket for execution events.
On a quiet connection it cross-checks `/queue` + `/history`
before giving up.

| Env | Default | Notes |
|---|---|---|
| `WEBSOCKET_INITIAL_TIMEOUT`        | `60`  | Seconds to wait for the first WS message after submitting the workflow. Bump for cold loads where the first inference step takes longer (large diffusion models on a cold GPU). |
| `WEBSOCKET_MESSAGE_TIMEOUT`        | `120` | Seconds between subsequent WS messages before falling back to queue/history polling. |
| `WEBSOCKET_MAX_NO_MESSAGE_RETRIES` | `3`   | When polling falls through too, fail the job after this many consecutive timeouts. |
| `WEBSOCKET_MAX_RECONNECTS`         | `10`  | When the WS closes mid-job and the job is still running on ComfyUI's side, how many times to reconnect before giving up. |

### Worker pools

| Env | Default | Notes |
|---|---|---|
| `PREPROCESS_WORKERS`  | `2`   | Concurrent preprocess tasks (URL fetch + workflow modification). |
| `GENERATION_WORKERS`  | `1`   | Ignored when `COMFYUI_BACKENDS` is set — one generation worker per backend. |
| `POSTPROCESS_WORKERS` | `2`   | Concurrent postprocess tasks (file move + S3 upload + webhook). |
| `MAX_QUEUE_SIZE`      | `100` | Inbound preprocess queue cap. Submits over this return HTTP 503 + `Retry-After: 5`. |

### Result envelope

| Env | Default | Notes |
|---|---|---|
| `INCLUDE_COMFYUI_RESPONSE` | `false` | When `true`, every Result includes the full ComfyUI `/history` blob in `comfyui_response`. Off by default to keep responses small. Per-request override via `X-Include-ComfyUI-Response: 1`. |

### Cache (request / response store)

The request and response stores back the wrapper's job state.
`memory` is a per-process dict — fine for a single wrapper
replica; results are lost when the process restarts. `redis`
shares state across replicas and survives wrapper restarts
within `CACHE_TTL`.

| Env | Default | Notes |
|---|---|---|
| `API_CACHE`      | `memory`    | `memory` or `redis`. |
| `API_CACHE_TTL`  | `21600`     | Seconds (6 h). How long results stay retrievable via `/result/{id}`. |
| `REDIS_HOST`     | `localhost` | Only when `API_CACHE=redis`. |
| `REDIS_PORT`     | `6379`      | |
| `REDIS_DB`       | `0`         | |
| `REDIS_PASSWORD` | (empty)     | Optional. |

`API_CACHE=redis` requires the optional `redis` package (already
in `requirements.txt`). The wrapper passes `REDIS_HOST` /
`REDIS_PORT` / `REDIS_DB` / `REDIS_PASSWORD` through to aiocache;
keys are namespaced `request_store:` and `response_store:`.

### Inline base64 output (optional)

When opted in, the postprocess worker reads each generated
file and embeds it as base64 under `output[*].data` (with a
sniffed `mimetype`). Useful for local dev (no S3 setup), small-
image pipelines that don't want a separate fetch round-trip,
and webhook consumers who otherwise get a useless internal
`local_path`.

Two ways to opt in (per-request):

- **Body field**: `"input": {"return_outputs_as_base64": true, ...}`.
- **Header**: `X-Return-Outputs-As-Base64: 1`. The header sets
  the body field server-side, so the postprocess flow is
  identical either way. Useful for `curl` and shell pipelines.

| Env | Default | Notes |
|---|---|---|
| `OUTPUT_BASE64_MAX_BYTES` | `10485760` (10 MB) | Per-file cap. Files larger than this are skipped with `output[*].error` set; the rest still inline. |

Base64 inflates payload size ~33%. The encoded `data` field
sits in the response store for `CACHE_TTL` (default 6 h), so
sustained traffic at large image sizes can chew memory — bump
`API_CACHE=redis` if that's a concern, and tune `CACHE_TTL`.

Coexists with S3: when both are configured, `data` and `url`
are both populated and the customer picks.

### S3 upload (optional)

When configured, `output[*]` files are uploaded after generation
and `output[*].url` is populated.

| Env | Notes |
|---|---|
| `S3_ACCESS_KEY_ID`     | Required to enable. |
| `S3_SECRET_ACCESS_KEY` | Required to enable. |
| `S3_BUCKET_NAME`       | Required to enable. |
| `S3_ENDPOINT_URL`      | Optional — for S3-compatible providers (R2, B2, MinIO). Leave unset for AWS S3. |
| `S3_REGION`            | |
| `S3_CONNECT_TIMEOUT`   | Default `60`. Seconds. |
| `S3_CONNECT_ATTEMPTS`  | Default `3`. |

A per-request `input.s3` block in the payload overrides these
defaults for that single job.

### Webhook (optional)

| Env | Notes |
|---|---|
| `WEBHOOK_URL`     | Default webhook endpoint; per-request `input.webhook.url` overrides. |
| `WEBHOOK_SECRET`  | Optional. When set, the wrapper signs every outgoing webhook with HMAC-SHA256 over the body bytes and sends `X-Webhook-Signature: sha256=<hex>`. Per-request `input.webhook.secret` overrides. See [§ Verifying a webhook signature](#verifying-a-webhook-signature). |
| `WEBHOOK_TIMEOUT` | Default `30`. Seconds. |

### Misc

| Env | Default | Notes |
|---|---|---|
| `DEBUG` | `false` | When `true`, prints a config summary at startup. |

## Multi-backend

The wrapper can drive N ComfyUI processes for parallelism — one
per GPU on a multi-GPU host, or several per GPU when the model
fits multiple times in VRAM.

```
                    ┌── ComfyUI on :8188 (CUDA_VISIBLE_DEVICES=0)
wrapper :8000  ─┬───┼── ComfyUI on :8189 (CUDA_VISIBLE_DEVICES=1)
                │   ├── ComfyUI on :8190 (CUDA_VISIBLE_DEVICES=2)
                │   └── ComfyUI on :8191 (CUDA_VISIBLE_DEVICES=3)
                generation_queue (one consumer per backend)
```

Set `COMFYUI_BACKENDS` to a comma-separated list of bases:

```bash
COMFYUI_BACKENDS="http://127.0.0.1:8188,http://127.0.0.1:8189"
```

The wrapper spawns one generation worker per entry; N consumers
reading from the shared `generation_queue` form a natural pool —
the queue itself routes each job to the first idle worker.

When `COMFYUI_BACKENDS` is unset, the wrapper falls back to the
single-backend `COMFYUI_API_BASE`. Existing single-backend
deployments need no changes.

Each ComfyUI process is independently health-checked, gets its
own GPU-fault latch, and runs OOM recovery (`/free`) in
isolation. Cancellation routes to the right backend
automatically — the worker that holds an in-flight job is the
one that posts `/api/interrupt`.

The wrapper does not start the ComfyUI processes itself —
operator's responsibility (typically supervisord, systemd, or a
process supervisor that reads container env). Pin each ComfyUI
to its GPU via `CUDA_VISIBLE_DEVICES`.

## Webhook

When `input.webhook.url` is set on the request (or the
`WEBHOOK_URL` env is set globally), the wrapper POSTs a summary
to that URL after postprocess finishes. Body shape:

```json
{
  "id":      "request-uuid",
  "status":  "completed",
  "message": "Processing complete.",
  "output":  [ /* same as Result.output */ ],
  "timings": { "preprocess_ms": 12, "generation_ms": 18412, "postprocess_ms": 743 },
  "extra":   { /* whatever the customer put in webhook.extra_params */ }
}
```

Notes:

- **HMAC-SHA256 signing** — opt-in via a shared secret, off by
  default. Set `WEBHOOK_SECRET` (env, applies globally) or
  per-request `input.webhook.secret` (overrides). When set, the
  wrapper sends `X-Webhook-Signature: sha256=<hex>` computed
  over the exact JSON body bytes the consumer receives. Per-
  request `secret` takes precedence; supply an empty
  per-request `secret` *together with* a per-request webhook
  block to opt this request out of signing while leaving the
  env default in place for other requests.
- **Fire-and-forget delivery.** A single attempt with a 30 s
  timeout, no retry. Failures are logged but don't affect the
  job. Webhook consumers should be idempotent (the same `id`
  may not be delivered twice in normal operation, but a flaky
  network can produce ambiguous outcomes).
- **No `comfyui_response`.** The full ComfyUI history blob is
  not in the webhook body — typically several KB and most
  consumers don't want it. Fetch it via `GET /result/{id}` if
  you need it, or set `INCLUDE_COMFYUI_RESPONSE=true`.
- **`extra` is namespaced.** Customer-supplied `extra_params`
  land under the `extra` key, NOT merged into the top level —
  so `extra_params: {"status": "anything"}` can't clobber the
  wrapper's `status` field. The signature covers the full body
  including `extra`.

### Verifying a webhook signature

Read the raw request body (NOT the parsed JSON dict — body bytes
matter) and recompute the HMAC. Use a constant-time comparison.

**Python (FastAPI / Flask):**

```python
import hmac, hashlib

WEBHOOK_SECRET = b"shared-secret"

def verify(raw_body: bytes, signature_header: str) -> bool:
    if not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(WEBHOOK_SECRET, raw_body, hashlib.sha256).hexdigest()
    received = signature_header.split("=", 1)[1]
    return hmac.compare_digest(received, expected)

# In a FastAPI route:
async def handler(request: Request):
    raw = await request.body()
    sig = request.headers.get("x-webhook-signature", "")
    if not verify(raw, sig):
        raise HTTPException(401, "Invalid webhook signature")
    payload = json.loads(raw)
    ...
```

**Node (Express):**

```javascript
const crypto = require('crypto');

function verify(rawBody, signatureHeader, secret) {
  if (!signatureHeader || !signatureHeader.startsWith('sha256=')) return false;
  const expected = crypto.createHmac('sha256', secret).update(rawBody).digest('hex');
  const received = signatureHeader.slice(7);
  // Use timingSafeEqual to avoid timing-attack leakage.
  const a = Buffer.from(expected, 'hex');
  const b = Buffer.from(received, 'hex');
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}
```

The signature is over the wire bytes, so your handler must read
the raw body before any JSON-parsing middleware mutates it
(Express needs `express.raw({type: 'application/json'})`).

## Health endpoint

`GET /health` walks every backend and returns 200 / 503 + a
JSON body:

```json
{
  "status":           "healthy",
  "cache_type":       "memory",
  "queues":           {"preprocess": 0, "generation": 0, "postprocess": 0},
  "backends_healthy": "2/2",
  "gpu":              {"unrecoverable": false, "reason": ""},
  "backends": [
    {
      "base":         "http://127.0.0.1:8188",
      "http_ok":      true,
      "websocket_ok": true,
      "system_stats": { /* ComfyUI's /system_stats output */ },
      "gpu":          {"unrecoverable": false, "reason": ""},
      "healthy":      true
    },
    { /* ... per backend ... */ }
  ]
}
```

The endpoint returns 503 when:

- Any backend's HTTP `/system_stats` doesn't return 200.
- Any backend's WebSocket connect fails.
- Any backend has latched a non-OOM CUDA fault (illegal memory
  access, device-side assert, misaligned address, etc.) — these
  don't survive ComfyUI's `/free`, so the wrapper marks the
  backend permanently unhealthy and the orchestrator should
  replace the pod.

OOM is **not** a health failure — the wrapper calls the affected
backend's `/free` to unload models and continues. The job that
hit OOM still fails (we don't retry inside the wrapper) but
subsequent jobs work.

## Reliability

A short list of failure modes the wrapper handles for you, in
case you've hit them in plain ComfyUI:

- **WebSocket goes silent mid-job.** The wrapper cross-checks
  `/queue` + `/history` and reconnects up to
  `WEBSOCKET_MAX_RECONNECTS` times if the job is still alive on
  ComfyUI's side. Without this, a transient WS close fails the
  job even when ComfyUI is still happily running it.
- **ComfyUI prompt-cache returns empty outputs.** ComfyUI
  assigns a *new* prompt_id on cache hit but doesn't copy the
  original outputs into the new history entry. The wrapper
  detects this (empty `outputs` map on a "completed" job),
  clears the stale entry from `/history`, and re-submits to
  force a fresh execution.
- **OOM mid-generation.** The wrapper detects "out of memory"
  in the error path, posts to ComfyUI's `/free` to unload all
  models, and keeps the pod alive. The OOM job itself is failed.
- **CUDA-unrecoverable faults.** Distinct from OOM: illegal
  memory access, device-side assert, etc. The wrapper latches
  this on the affected backend and `/health` flips to 503. Your
  orchestrator should treat that as "replace the pod".
- **Workflow `output[*]` references a stale path.** Postprocess
  prefers ComfyUI's `files` array (this run's actual outputs)
  over `images` (which can carry stale references from a cached
  prior run) when both are present.
- **Wrapper starts before ComfyUI is listening.** First-job
  startup probes ComfyUI's HTTP and WebSocket endpoints before
  the first POST.
- **Inbound saturation.** When the preprocess queue is full,
  submits return 503 + `Retry-After: 5` instead of stacking
  forever. Lets the caller back off.

## Workflow modifiers

Two ways to specify a workflow:

### Raw workflow

Send the API-format workflow directly:

```json
{
  "input": {
    "workflow_json": {
      "10": {
        "inputs": { "image": "https://example.com/input.jpg", "upload": "image" },
        "class_type": "LoadImage"
      }
      // … rest of workflow
    }
  }
}
```

URLs anywhere in the workflow are downloaded automatically (see
[§ Automatic URL fetch](#automatic-url-fetch)).

### Curated modifier classes

Bundle a workflow and the parameter-substitution logic into a
Python class derived from `BaseModifier`:

```python
# modifiers/image2image.py
from modifiers.basemodifier import BaseModifier
import random


class Image2Image(BaseModifier):
    WORKFLOW_JSON = "workflows/image2image.json"

    async def apply_modifications(self):
        self.workflow["3"]["inputs"]["seed"]  = await self.modify_workflow_value(
            "seed", random.randint(0, 2**32))
        self.workflow["6"]["inputs"]["text"]  = await self.modify_workflow_value(
            "prompt", "")
        self.workflow["10"]["inputs"]["image"] = await self.modify_workflow_value(
            "input_image", "https://example.com/default.jpg")
        await super().apply_modifications()      # runs the URL-download pass
```

Customer payload:

```json
{
  "input": {
    "modifier": "Image2Image",
    "modifications": {
      "prompt": "a beautiful sunset",
      "seed": 12345,
      "input_image": "https://example.com/photo.jpg"
    }
  }
}
```

`workflow_json` and `modifier` are mutually exclusive — supply
exactly one. `modifications` is silently ignored without
`modifier`.

### Automatic URL fetch

`BaseModifier.replace_workflow_urls()` walks the workflow before
submission. Any string that parses as an HTTP/HTTPS URL is:

1. Downloaded (one-shot httpx GET).
2. MD5-hashed; cached in ComfyUI's `input/` directory keyed by
   the hash so repeat submissions don't re-download.
3. MIME-type-detected via `python-magic` for the right file
   extension.
4. Replaced in the workflow with the local filename.

Both raw-workflow and modifier-mode submissions go through this
pass.

## Error handling

| HTTP | When |
|---|---|
| `200` | Sync result delivered, or `/result/{id}` lookup succeeded. |
| `202` | Async submit accepted (returns the request id). |
| `400` | Invalid JSON / Pydantic validation failure. |
| `404` | `/result/{id}` or `/cancel/{id}` for an unknown id. |
| `499` | `/generate/sync` only — client closed the connection. The job is marked `cancelled`. |
| `500` | Unexpected wrapper error. |
| `503` | Inbound queue saturated (`Retry-After: 5`), or `/health` failed (one or more backends down / GPU latched). |

A Result envelope with `status: "failed"` is **not** an HTTP
error — the request was processed, the workflow execution
failed. Inspect `message` and `comfyui_response.<id>.status`
for the engine-side reason.

## Architecture

```
                    ┌─ preprocess_worker × N (URL fetch, modifier apply)
   /generate{,/sync,/stream}
   ──▶ Pydantic validate ──▶ preprocess_queue
                                 │
                                 ▼
                              generation_worker × len(COMFYUI_BACKENDS)
                              (POST /prompt → WS wait → /history)
                                 │
                                 ▼
                              postprocess_worker × N
                              (move outputs, S3 upload, webhook)
                                 │
                                 ▼
                              response_store ──▶ /result, /generate/sync resolve, SSE final_result
```

The three queues are in-process `asyncio.Queue`s. The Result
envelope flows through the response store (`memory` or
`redis` per `API_CACHE`); set Redis if you want results to
survive a wrapper process restart.

## Development

```bash
pip install -r requirements-dev.txt
pytest                # unit tests for the bits that don't need ComfyUI
uvicorn main:app --reload --port 8000
```

Tests cover the OOM/CUDA error classifiers, the cached-output
validation, and the multi-backend config parser. They don't
exercise the WebSocket path against a real ComfyUI — that's
left for integration testing in a downstream consumer.

## Limitations to know about

- Cancellation is **soft**, and only effective during preprocess
  and generation. Once a job reaches postprocess (file moves, S3
  upload, webhook delivery) cancel is a no-op — the postprocess
  worker doesn't check the cancel marker. ComfyUI itself may also
  take a few seconds to stop sampling after an `/api/interrupt`,
  so a cancel that arrives mid-generation may complete *after* an
  output has already been written.
- The cache (`memory` or `redis`) holds Results for `CACHE_TTL`
  seconds. Don't rely on `/result/{id}` after that window.
- `output[*].local_path` is internal to the wrapper container.
  If you need files reachable externally, set up S3 (then `url`
  is populated) or run the wrapper next to whatever serves the
  files (a sidecar, a shared volume, etc.). The wrapper does not
  serve output files directly.
- Webhook delivery is fire-and-forget — no retry on failure
  beyond the underlying httpx attempt. Consume webhooks
  idempotently.
