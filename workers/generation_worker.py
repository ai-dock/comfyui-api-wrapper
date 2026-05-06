# generation_worker
import asyncio
import aiohttp
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from config import (
    COMFYUI_API_PROMPT,
    COMFYUI_API_QUEUE,
    COMFYUI_API_HISTORY,
    COMFYUI_API_INTERRUPT,
    COMFYUI_API_FREE,
    COMFYUI_API_WEBSOCKET,
    WEBSOCKET_INITIAL_TIMEOUT,
    WEBSOCKET_MESSAGE_TIMEOUT,
    WEBSOCKET_MAX_NO_MESSAGE_RETRIES,
    WEBSOCKET_MAX_RECONNECTS,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Module-level GPU-unrecoverable state.
#
# When the wrapper detects a GPU fault that ComfyUI's `/free` cannot
# recover from (illegal memory access, misaligned address, device-side
# assert, etc.), we latch the reason here. The wrapper's `/health`
# endpoint reads it and reports 503 so an external health monitor /
# orchestrator can replace the pod on a clean host.
#
# OOM is NOT considered unrecoverable — it gets handled inline via
# `_attempt_oom_recovery()` calling ComfyUI's `/free`.
# ----------------------------------------------------------------------
_GPU_STATE: Dict[str, Any] = {"unrecoverable": False, "reason": ""}


def mark_gpu_unrecoverable(reason: str) -> None:
    """Latch the GPU-unrecoverable flag. Idempotent — preserves the
    first reason recorded so the operator sees the originating fault
    rather than a downstream one."""
    if not _GPU_STATE["unrecoverable"]:
        _GPU_STATE["unrecoverable"] = True
        _GPU_STATE["reason"] = reason or "unspecified"
        logger.error(f"GPU latched as unrecoverable: {reason}")


def get_gpu_state() -> Dict[str, Any]:
    return dict(_GPU_STATE)


class GenerationWorker:
    """
    Send payload to ComfyUI and await completion using WebSocket
    """
    def __init__(self, worker_id, kwargs):
        self.worker_id = worker_id
        self.preprocess_queue = kwargs["preprocess_queue"]
        self.generation_queue = kwargs["generation_queue"]
        self.postprocess_queue = kwargs["postprocess_queue"]
        self.request_store = kwargs["request_store"]
        self.response_store = kwargs["response_store"]

        # Configuration
        self.max_wait_time = 3600  # 1 hour maximum wait
        self.ws_url = COMFYUI_API_WEBSOCKET
        self.client_id = f"worker_{worker_id}_{datetime.now().timestamp()}"

        # Under supervisord (or any process supervisor co-launching the
        # wrapper alongside ComfyUI) the wrapper can come up before
        # ComfyUI is listening; the first POST then fails. Latch
        # readiness on first job and skip the probe thereafter.
        self._comfy_ready = False

    async def work(self):
        logger.info(f"GenerationWorker {self.worker_id}: waiting for jobs")
        while True:
            # Get a task from the job queue
            request_id = await self.generation_queue.get()
            if request_id is None:
                # None is a signal that there are no more tasks
                break

            # Process the job
            logger.info(f"GenerationWorker {self.worker_id} processing job: {request_id}")

            try:
                # Get request and result from stores
                request = await self.request_store.get(request_id)
                result = await self.response_store.get(request_id)

                if not request:
                    raise Exception(f"Request {request_id} not found in store")
                if not result:
                    raise Exception(f"Result {request_id} not found in store")

                # Check for cancellation
                if result and getattr(result, 'status', '') == 'cancelled':
                    logger.info(f"PreprocessWorker {self.worker_id} skipping cancelled job: {request_id} - jumping to postprocess")
                    await self.postprocess_queue.put(request_id)
                    self.generation_queue.task_done()
                    continue

                # First-job startup gate. ComfyUI takes a few seconds
                # to come up on a cold pod — wait for HTTP + WS to be
                # reachable before posting the workflow.
                if not self._comfy_ready:
                    await self.wait_for_comfy_ready()
                    self._comfy_ready = True

                # Submit workflow to ComfyUI
                comfyui_job_id = await self.post_workflow(request)
                logger.info(f"Submitted job {request_id} to ComfyUI as {comfyui_job_id}")

                # Update status to show generation started
                result.status = "generating"
                result.message = f"Generation started (ComfyUI job: {comfyui_job_id})"
                await self.response_store.set(request_id, result)

                # Check if job is already complete (cached result).
                # ComfyUI's prompt cache hashes inputs and returns the
                # original prompt's outputs — but in some node-graph
                # shapes the new prompt_id has an empty outputs map,
                # which produces silent successes with no usable
                # output references. Validate that any cached entry
                # actually carries output references; if not, clear
                # and re-submit.
                is_cached = await self.check_if_cached(comfyui_job_id)

                if is_cached:
                    logger.info(f"Job {comfyui_job_id} detected as cached — validating outputs")
                    cached_response = await self.get_result(comfyui_job_id)
                    if await self._validate_cached_outputs(cached_response, comfyui_job_id):
                        logger.info(f"Job {comfyui_job_id} cached with valid outputs")
                        execution_result = {
                            "prompt_id": comfyui_job_id,
                            "nodes_executed": [],
                            "progress_updates": [],
                            "completed": True,
                            "cached": True,
                            "error": None,
                        }
                    else:
                        logger.warning(
                            f"Cached entry for {comfyui_job_id} has no usable outputs — "
                            f"clearing ComfyUI history and re-submitting"
                        )
                        await self._clear_comfyui_history(comfyui_job_id)
                        comfyui_job_id = await self.post_workflow(request)
                        logger.info(f"Re-submitted job {request_id} to ComfyUI as {comfyui_job_id}")
                        result.comfyui_job_id = comfyui_job_id
                        await self.response_store.set(request_id, result)
                        is_cached = False  # force the WS path

                if not is_cached:
                    # Wait for completion using WebSocket (reconnects
                    # automatically if the WS drops mid-job).
                    execution_result = await self.wait_for_completion_websocket(
                        comfyui_job_id,
                        request_id
                    )

                # Get the final result from ComfyUI history
                comfyui_response = await self.get_result(comfyui_job_id)
                logger.info(f"Retrieved ComfyUI result for {request_id}")
                logger.debug(f"ComfyUI response structure: {json.dumps(comfyui_response, indent=2)[:500]}...")  # First 500 chars

                # Update result with success
                result.status = "generated"
                result.message = "Generation complete. Queued for post-processing."
                result.comfyui_response = comfyui_response
                # Store execution details in the comfyui_response if needed
                if execution_result:
                    # Merge execution details into the response
                    if isinstance(result.comfyui_response, dict):
                        result.comfyui_response["execution_details"] = execution_result
                await self.response_store.set(request_id, result)

                # Send for post-processing
                await self.postprocess_queue.put(request_id)
                logger.info(f"GenerationWorker {self.worker_id} completed job: {request_id}")

            except Exception as e:
                error_message = str(e)
                logger.error(f"GenerationWorker {self.worker_id} failed job {request_id}: {e}")

                # OOM is recoverable — call ComfyUI's /free to unload
                # models and reset VRAM, then move on. The job itself
                # is still failed (we won't retry it here), but the
                # next request to this pod will start clean.
                if _detect_oom_error(error_message):
                    logger.warning(f"OOM detected for {request_id} — calling /free")
                    if await _attempt_oom_recovery():
                        logger.info("OOM recovery succeeded; pod stays healthy")
                    else:
                        logger.error("OOM recovery failed; pod stays alive but next job may also OOM")
                else:
                    # Non-OOM CUDA fault → mark the GPU unrecoverable.
                    # /health will then return 503 so an external
                    # health monitor can replace the pod.
                    cuda_reason = _detect_cuda_unrecoverable_reason(error_message)
                    if cuda_reason:
                        mark_gpu_unrecoverable(cuda_reason)

                try:
                    # Update result to show failure
                    result = await self.response_store.get(request_id)
                    if result:
                        # Don't overwrite an already-cancelled status.
                        if getattr(result, 'status', '') != 'cancelled':
                            result.status = "failed"
                            result.message = f"Generation failed: {error_message}"
                            await self.response_store.set(request_id, result)

                    # Send job to postprocess for cleanup
                    await self.postprocess_queue.put(request_id)

                except Exception as store_error:
                    logger.error(f"Failed to update result store for {request_id}: {store_error}")

            finally:
                # Mark the job as complete
                self.generation_queue.task_done()

        logger.info(f"GenerationWorker {self.worker_id} finished")

    async def wait_for_comfy_ready(self, max_wait_seconds: int = 120) -> None:
        """Block until ComfyUI's HTTP and WebSocket endpoints are
        reachable. Mitigates the supervisord race where the wrapper
        starts before ComfyUI."""
        start = asyncio.get_event_loop().time()
        attempt = 0
        http_timeout = aiohttp.ClientTimeout(total=2.0)

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > max_wait_seconds:
                raise Exception(f"ComfyUI not ready after {max_wait_seconds}s")

            attempt += 1
            http_ok = False
            ws_ok = False

            try:
                async with aiohttp.ClientSession(timeout=http_timeout) as session:
                    async with session.get(COMFYUI_API_HISTORY) as resp:
                        # 200 means the history endpoint replied;
                        # 404 means ComfyUI is up but the URL form is
                        # slightly different — both indicate listening.
                        http_ok = resp.status in (200, 404)
            except Exception:
                http_ok = False

            try:
                async with aiohttp.ClientSession(timeout=http_timeout) as session:
                    async with session.ws_connect(self.ws_url, params={"clientId": "healthcheck"}) as ws:
                        ws_ok = True
                        await ws.close()
            except Exception:
                ws_ok = False

            if http_ok and ws_ok:
                logger.info(f"ComfyUI ready after {elapsed:.1f}s ({attempt} probe(s))")
                return

            backoff = min(0.5 * (2 ** min(attempt, 5)), 3.0)
            logger.info(
                f"ComfyUI not ready (http_ok={http_ok}, ws_ok={ws_ok}); "
                f"retry {attempt} in {backoff:.1f}s"
            )
            await asyncio.sleep(backoff)

    async def post_workflow(self, request) -> str:
        """Submit workflow to ComfyUI API.

        Retries up to 5 times with linear backoff on transient
        failures. ComfyUI's /prompt occasionally 5xx's during model
        loads; the retry typically lands.
        """
        payload = {
            "prompt": request.input.workflow_json,
            "client_id": self.client_id  # Use our worker's client ID
        }

        headers = {
            'Content-Type': 'application/json'
        }

        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout per attempt

        max_attempts = 5
        base_delay_seconds = 1.0
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    logger.debug(f"Posting workflow to {COMFYUI_API_PROMPT} (attempt {attempt}/{max_attempts})")
                    logger.debug(
                        f"Workflow keys: "
                        f"{list(request.input.workflow_json.keys()) if isinstance(request.input.workflow_json, dict) else 'not a dict'}"
                    )

                    async with session.post(
                        COMFYUI_API_PROMPT,
                        data=json.dumps(payload),
                        headers=headers
                    ) as response:

                        response_text = await response.text()
                        logger.debug(f"ComfyUI API response status: {response.status}")
                        logger.debug(f"ComfyUI API response: {response_text[:500]}...")  # First 500 chars

                        if response.status >= 500:
                            # Transient — retry.
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"ComfyUI 5xx: {response_text[:200]}"
                            )
                        if response.status >= 400:
                            # 4xx is the workflow's fault — don't retry.
                            response_data = json.loads(response_text) if response_text else {}
                            if "node_errors" in response_data:
                                error_details = json.dumps(response_data["node_errors"], indent=2)
                                raise Exception(f"ComfyUI node errors: {error_details}")
                            if "error" in response_data:
                                raise Exception(f"ComfyUI error: {response_data['error']}")
                            raise Exception(f"ComfyUI {response.status}: {response_text[:200]}")

                        response_data = json.loads(response_text)

                        if "prompt_id" in response_data:
                            return response_data["prompt_id"]
                        if "node_errors" in response_data:
                            error_details = json.dumps(response_data["node_errors"], indent=2)
                            raise Exception(f"ComfyUI node errors: {error_details}")
                        if "error" in response_data:
                            raise Exception(f"ComfyUI error: {response_data['error']}")
                        raise Exception(f"Unexpected response from ComfyUI: {response_data}")

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_error = e
                if attempt < max_attempts:
                    delay = base_delay_seconds * attempt
                    logger.warning(
                        f"Transient error posting workflow (attempt {attempt}/{max_attempts}): {e}; "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                break
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON response from ComfyUI: {e}")

        raise Exception(f"Failed to post workflow after {max_attempts} attempts: {last_error}")

    async def check_if_cached(self, comfyui_job_id: str) -> bool:
        """Check if job is already complete (in history)."""
        await asyncio.sleep(0.5)  # Give ComfyUI a moment to process

        timeout = aiohttp.ClientTimeout(total=5)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{COMFYUI_API_HISTORY}/{comfyui_job_id}"
                async with session.get(url) as response:
                    if response.status == 200:
                        history_data = await response.json()
                        # If we get non-empty data, the job is complete
                        if history_data and history_data != {}:
                            logger.info(f"Job {comfyui_job_id} found in history (cached)")
                            return True
            return False
        except Exception as e:
            logger.debug(f"Error checking cache status: {e}")
            return False

    async def check_if_running(self, comfyui_job_id: str) -> bool:
        """Check if a job is currently running OR pending in ComfyUI's queue."""
        timeout = aiohttp.ClientTimeout(total=5)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(COMFYUI_API_QUEUE) as response:
                    if response.status != 200:
                        return False
                    data = await response.json()
                    # Queue shape: {"queue_running": [[<n>, "<id>", ...], ...],
                    #               "queue_pending": [[<n>, "<id>", ...], ...]}
                    for bucket in ("queue_running", "queue_pending"):
                        for entry in data.get(bucket) or []:
                            # The prompt_id sits at index 1 in each entry tuple.
                            if isinstance(entry, list) and len(entry) >= 2 and entry[1] == comfyui_job_id:
                                return True
            return False
        except Exception as e:
            logger.debug(f"Error checking queue status: {e}")
            return False

    async def _validate_cached_outputs(
        self, comfyui_response: Optional[dict], comfyui_job_id: str
    ) -> bool:
        """Confirm a cached ComfyUI response actually has output
        references the postprocess can use.

        ComfyUI's prompt cache assigns a NEW prompt_id but does not
        copy the original prompt's output references — the new
        history entry's `outputs` is `{}`. The wrapper sees
        "completed" but produces nothing. Detect that case so the
        caller can clear the cache entry and force a fresh
        execution.
        """
        if not comfyui_response or not isinstance(comfyui_response, dict):
            return False
        prompt_data = comfyui_response.get(comfyui_job_id) or {}
        outputs = prompt_data.get("outputs") if isinstance(prompt_data, dict) else None
        if not outputs or not isinstance(outputs, dict):
            return False
        # Any node carrying a non-empty list under any output type counts.
        for node_id, node_outputs in outputs.items():
            if not isinstance(node_outputs, dict):
                continue
            for output_type, items in node_outputs.items():
                if isinstance(items, list) and items:
                    return True
        return False

    async def _clear_comfyui_history(self, comfyui_job_id: str) -> None:
        """Drop a stale prompt from ComfyUI's history so re-submitting
        the same payload misses the cache and runs fresh."""
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # ComfyUI accepts {"clear": true, "delete": [<id>...]}
                # against /history.
                async with session.post(
                    COMFYUI_API_HISTORY,
                    json={"delete": [comfyui_job_id]},
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status in (200, 204):
                        logger.info(f"Cleared {comfyui_job_id} from ComfyUI history")
                        return
                    body = await response.text()
                    logger.warning(
                        f"Could not clear {comfyui_job_id} from history "
                        f"(status={response.status}): {body[:200]}"
                    )
        except Exception as e:
            logger.warning(f"Error clearing history for {comfyui_job_id}: {e}")

    async def wait_for_completion_websocket(self, comfyui_job_id: str, request_id: str) -> Dict[str, Any]:
        """Wait for ComfyUI job completion via WebSocket.

        Reconnects automatically if the WS closes while the job is
        still in ComfyUI's queue or running. Long workflows can see
        the WS go silent for tens of seconds; without a reconnect
        the wrapper would time out while the job was actually still
        progressing.
        """
        execution_result = {
            "prompt_id": comfyui_job_id,
            "nodes_executed": [],
            "progress_updates": [],
            "completed": False,
            "error": None,
        }

        start_time = asyncio.get_event_loop().time()
        reconnect_count = 0
        max_reconnects = WEBSOCKET_MAX_RECONNECTS

        while reconnect_count <= max_reconnects:
            try:
                completed = await self._ws_listen_loop(
                    comfyui_job_id, request_id, execution_result, start_time
                )
                if completed:
                    return execution_result

                # _ws_listen_loop returned False — the WS closed. Decide
                # whether to reconnect based on whether the job is
                # still alive on ComfyUI's side.
                if await self.check_if_cached(comfyui_job_id):
                    logger.info(f"Job {comfyui_job_id} completed (post-close history check)")
                    execution_result["completed"] = True
                    return execution_result

                still_running = False
                try:
                    still_running = await self.check_if_running(comfyui_job_id)
                except Exception:
                    pass

                if still_running:
                    reconnect_count += 1
                    logger.info(
                        f"WebSocket closed but job {comfyui_job_id} still running — "
                        f"reconnecting ({reconnect_count}/{max_reconnects})"
                    )
                    await asyncio.sleep(2)
                    continue

                # Final history check before giving up — race window
                # where the job finished AND fell out of the queue
                # between the close and the queue check.
                await asyncio.sleep(1)
                if await self.check_if_cached(comfyui_job_id):
                    logger.info(f"Job {comfyui_job_id} completed (final post-close check)")
                    execution_result["completed"] = True
                    return execution_result

                raise Exception(
                    f"WebSocket closed and job {comfyui_job_id} not in queue or history"
                )

            except asyncio.TimeoutError:
                logger.warning(f"WebSocket overall timeout for job {comfyui_job_id}")
                await self.cancel_comfyui_job(comfyui_job_id)
                raise Exception(f"WebSocket timeout for job {comfyui_job_id}")
            except aiohttp.ClientError as e:
                # Connection-level failure — same logic as a clean
                # close: check if the job lives on, reconnect if so.
                if await self.check_if_cached(comfyui_job_id):
                    logger.info(f"Job {comfyui_job_id} completed despite ClientError")
                    execution_result["completed"] = True
                    return execution_result
                still_running = False
                try:
                    still_running = await self.check_if_running(comfyui_job_id)
                except Exception:
                    pass
                if still_running:
                    reconnect_count += 1
                    logger.info(
                        f"ClientError but job still running — reconnecting "
                        f"({reconnect_count}/{max_reconnects}): {e}"
                    )
                    await asyncio.sleep(2)
                    continue
                await self.cancel_comfyui_job(comfyui_job_id)
                raise Exception(f"WebSocket connection error: {e}")

        # max_reconnects exceeded
        if not execution_result["completed"]:
            raise Exception(
                f"Max WebSocket reconnects ({max_reconnects}) reached for job {comfyui_job_id}"
            )
        return execution_result

    async def _ws_listen_loop(
        self,
        comfyui_job_id: str,
        request_id: str,
        execution_result: Dict[str, Any],
        start_time: float,
    ) -> bool:
        """One WebSocket connection's listen loop.

        Returns True if the job completed within this connection;
        False if the connection closed cleanly (the caller decides
        whether to reconnect). Raises on overall timeout, errors,
        or external cancellation.
        """
        timeout = aiohttp.ClientTimeout(total=self.max_wait_time)
        initial_timeout = WEBSOCKET_INITIAL_TIMEOUT
        message_timeout = WEBSOCKET_MESSAGE_TIMEOUT
        max_no_message_retries = WEBSOCKET_MAX_NO_MESSAGE_RETRIES
        no_message_retry_count = 0

        async with aiohttp.ClientSession(timeout=timeout) as session:
            logger.info(f"Connecting to ComfyUI WebSocket at {self.ws_url}")
            async with session.ws_connect(
                self.ws_url,
                params={"clientId": self.client_id}
            ) as ws:
                logger.info(f"WebSocket connected for job {comfyui_job_id}")

                last_update_time = asyncio.get_event_loop().time()
                last_message_time = start_time
                last_cancellation_check = start_time

                while True:
                    try:
                        timeout_duration = (
                            initial_timeout if last_message_time == start_time
                            else message_timeout
                        )

                        msg = await asyncio.wait_for(
                            ws.receive(),
                            timeout=timeout_duration
                        )

                        last_message_time = asyncio.get_event_loop().time()
                        no_message_retry_count = 0

                        current_time = asyncio.get_event_loop().time()
                        if current_time - last_cancellation_check > 5.0:
                            if await self._check_if_cancelled(request_id):
                                logger.info(f"Job {request_id} cancelled — aborting WebSocket")
                                await self.cancel_comfyui_job(comfyui_job_id)
                                raise Exception(f"Job {request_id} was cancelled during generation")
                            last_cancellation_check = current_time

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                message_type = data.get("type")
                                logger.debug(f"WebSocket message type: {message_type}")

                                if data.get("data", {}).get("prompt_id") == comfyui_job_id:

                                    if message_type == "execution_start":
                                        logger.debug(f"Execution started for {comfyui_job_id}")
                                        await self._update_progress(
                                            request_id,
                                            "Execution started..."
                                        )

                                    elif message_type == "execution_cached":
                                        nodes = data.get("data", {}).get("nodes", [])
                                        logger.debug(f"Using cached results for nodes: {nodes}")
                                        execution_result["nodes_executed"].extend(nodes)

                                    elif message_type == "executing":
                                        node = data.get("data", {}).get("node")
                                        if node:
                                            logger.debug(f"Executing node: {node}")
                                            execution_result["nodes_executed"].append(node)
                                            await self._update_progress(
                                                request_id,
                                                f"Processing node: {node}"
                                            )
                                        elif data.get("data", {}).get("node") is None:
                                            # node = None means execution complete
                                            logger.info(f"Execution complete for {comfyui_job_id}")
                                            execution_result["completed"] = True
                                            return True

                                    elif message_type == "progress":
                                        progress_data = data.get("data", {})
                                        value = progress_data.get("value", 0)
                                        max_value = progress_data.get("max", 100)
                                        progress_pct = (value / max_value * 100) if max_value > 0 else 0
                                        progress_msg = f"Progress: {progress_pct:.1f}% ({value}/{max_value})"
                                        logger.debug(f"Progress update: {progress_msg}")
                                        execution_result["progress_updates"].append({
                                            "time": asyncio.get_event_loop().time() - start_time,
                                            "value": value,
                                            "max": max_value,
                                            "percentage": progress_pct,
                                        })
                                        current_time = asyncio.get_event_loop().time()
                                        if current_time - last_update_time > 2:
                                            await self._update_progress(request_id, progress_msg)
                                            last_update_time = current_time

                                    elif message_type == "execution_error":
                                        error_data = data.get("data", {})
                                        error_msg = f"Execution error: {error_data}"
                                        logger.error(error_msg)
                                        execution_result["error"] = error_data
                                        raise Exception(error_msg)

                                    elif message_type == "executed":
                                        node = data.get("data", {}).get("node")
                                        output = data.get("data", {}).get("output")
                                        logger.debug(f"Node {node} executed successfully")
                                        logger.debug(f"Node output: {json.dumps(output, indent=2)[:500]}...")

                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse WebSocket message: {e}")
                                logger.debug(f"Raw message: {msg.data}")

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            raise Exception(f"WebSocket error: {ws.exception()}")

                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.warning("WebSocket closed — returning to reconnect logic")
                            return False

                    except asyncio.TimeoutError:
                        no_message_retry_count += 1
                        elapsed = asyncio.get_event_loop().time() - start_time
                        source = "initial" if last_message_time == start_time else "ongoing"
                        logger.warning(
                            f"WebSocket timeout ({source}) for {comfyui_job_id} "
                            f"(attempt {no_message_retry_count}/{max_no_message_retries}) "
                            f"after {elapsed:.1f}s"
                        )

                        try:
                            if await self.check_if_cached(comfyui_job_id):
                                logger.info(f"Job {comfyui_job_id} complete (history check)")
                                execution_result["completed"] = True
                                return True
                        except Exception as check_error:
                            logger.warning(f"Error checking history: {check_error}")

                        still_running = False
                        try:
                            still_running = await self.check_if_running(comfyui_job_id)
                        except Exception as check_error:
                            logger.warning(f"Error checking queue: {check_error}")

                        if still_running:
                            logger.info(
                                f"Job {comfyui_job_id} still running — continuing to wait "
                                f"(elapsed: {elapsed:.1f}s)"
                            )
                            no_message_retry_count = 0  # alive — reset
                            await asyncio.sleep(10)
                            continue

                        if no_message_retry_count >= max_no_message_retries:
                            logger.error(
                                f"Job {comfyui_job_id} not in queue or history after "
                                f"{max_no_message_retries} attempts ({elapsed:.1f}s elapsed)"
                            )
                            raise Exception(
                                f"Job {comfyui_job_id} disappeared from ComfyUI "
                                f"after {no_message_retry_count} attempts"
                            )

                        wait_time = min(5 * (2 ** (no_message_retry_count - 1)), 30)
                        logger.info(
                            f"Job not in queue, waiting {wait_time}s before retry "
                            f"{no_message_retry_count + 1}/{max_no_message_retries}"
                        )
                        await asyncio.sleep(wait_time)

                    # Check for overall timeout
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > self.max_wait_time:
                        raise Exception(
                            f"Timeout waiting for job {comfyui_job_id} after {elapsed:.1f}s"
                        )

        # Should not reach here.
        return False

    async def _update_progress(self, request_id: str, message: str):
        """Helper to update progress in the response store"""
        try:
            result = await self.response_store.get(request_id)
            if result:
                result.message = message
                await self.response_store.set(request_id, result)
        except Exception as e:
            logger.warning(f"Failed to update progress for {request_id}: {e}")

    async def get_result(self, comfyui_job_id: str) -> Optional[dict]:
        """Get the final result from ComfyUI history"""
        timeout = aiohttp.ClientTimeout(total=30)

        # Wait a moment for history to be updated
        await asyncio.sleep(0.5)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{COMFYUI_API_HISTORY}/{comfyui_job_id}"
                logger.debug(f"Fetching result from: {url}")

                async with session.get(url) as response:
                    response_text = await response.text()
                    logger.debug(f"History API status: {response.status}")

                    if response.status == 200:
                        history_data = json.loads(response_text)

                        # Check if we got actual data
                        if not history_data or history_data == {}:
                            logger.warning(f"Empty history response for job {comfyui_job_id}")
                            # Try the general history endpoint
                            return await self._get_result_from_general_history(comfyui_job_id)

                        logger.info(f"Retrieved ComfyUI history for job {comfyui_job_id}")
                        return history_data
                    else:
                        raise Exception(f"Failed to get result (status {response.status}): {response_text}")

        except asyncio.TimeoutError:
            raise Exception(f"Timeout getting result for job {comfyui_job_id}")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error getting result: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in result: {e}")

    async def _get_result_from_general_history(self, comfyui_job_id: str) -> Optional[dict]:
        """Fallback: Get result from general history endpoint"""
        timeout = aiohttp.ClientTimeout(total=30)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Try the general history endpoint
                url = COMFYUI_API_HISTORY.rstrip(f"/{comfyui_job_id}")
                logger.debug(f"Trying general history endpoint: {url}")

                async with session.get(url) as response:
                    if response.status == 200:
                        all_history = await response.json()

                        # Look for our job in the history
                        if comfyui_job_id in all_history:
                            logger.info(f"Found job {comfyui_job_id} in general history")
                            return {comfyui_job_id: all_history[comfyui_job_id]}
                        else:
                            logger.warning(f"Job {comfyui_job_id} not found in general history")
                            return {}
                    else:
                        return {}

        except Exception as e:
            logger.error(f"Failed to get result from general history: {e}")
            return {}

    async def _check_if_cancelled(self, request_id: str) -> bool:
        """Check if the job has been cancelled"""
        try:
            result = await self.response_store.get(request_id)
            return result and getattr(result, 'status', '') == 'cancelled'
        except Exception as e:
            logger.warning(f"Error checking cancellation status for {request_id}: {e}")
            return False

    async def cancel_comfyui_job(self, comfyui_job_id: str):
        """Cancel a running job in ComfyUI"""
        try:
            if not COMFYUI_API_INTERRUPT:
                logger.warning("COMFYUI_API_INTERRUPT not configured, cannot cancel job")
                return False

            payload = {
                "prompt_id": comfyui_job_id
            }

            headers = {
                'Content-Type': 'application/json'
            }

            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                cancel_url = COMFYUI_API_INTERRUPT

                async with session.post(
                    cancel_url,
                    data=json.dumps(payload),
                    headers=headers
                ) as response:

                    if response.status == 200:
                        logger.info(f"Successfully cancelled ComfyUI job {comfyui_job_id}")
                        return True
                    else:
                        response_text = await response.text()
                        logger.warning(
                            f"Failed to cancel ComfyUI job {comfyui_job_id}: "
                            f"HTTP {response.status} - {response_text}"
                        )
                        return False

        except Exception as e:
            logger.error(f"Error cancelling ComfyUI job {comfyui_job_id}: {e}")
            return False


# ----------------------------------------------------------------------
# Module-level error classifiers + recovery helpers.
# ----------------------------------------------------------------------

_OOM_TRIGGERS = (
    "out of memory",
    "cuda out of memory",
    "outofmemoryerror",
    "failed to allocate",
    "not enough memory",
    "memory allocation failed",
    "ran out of memory",
    "insufficient memory",
)

_CUDA_UNRECOVERABLE_TRIGGERS = (
    "unspecified launch failure",
    "device-side assert triggered",
    "illegal memory access",
    "an illegal memory access was encountered",
    "triton error [cuda]",
    "cuda runtime error",
    "cuda error: misaligned address",
    "cuda error: invalid device function",
    "cuda error: invalid configuration argument",
)


def _detect_oom_error(text: str) -> bool:
    """OOM is recoverable via ComfyUI's /free; classified separately
    from fatal CUDA faults."""
    if not text:
        return False
    lowered = text.lower()
    return any(trigger in lowered for trigger in _OOM_TRIGGERS)


def _detect_cuda_unrecoverable_reason(text: str) -> str:
    """Detect non-OOM CUDA faults that survive `/free` and require a
    pod restart. Returns a short reason string when matched, else "".
    OOM is excluded — that path is handled by `_attempt_oom_recovery`.
    """
    if not text:
        return ""
    if _detect_oom_error(text):
        return ""
    lowered = text.lower()
    for trigger in _CUDA_UNRECOVERABLE_TRIGGERS:
        if trigger in lowered:
            return trigger
    return ""


async def _attempt_oom_recovery() -> bool:
    """POST to ComfyUI's /free to unload all models and clear the
    CUDA cache. Returns True on success."""
    if not COMFYUI_API_FREE:
        logger.warning("COMFYUI_API_FREE not configured; cannot attempt OOM recovery")
        return False

    payload = {"unload_models": True, "free_memory": True}
    timeout = aiohttp.ClientTimeout(total=30)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            logger.info(f"OOM recovery: POST {COMFYUI_API_FREE}")
            async with session.post(
                COMFYUI_API_FREE,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    logger.info("OOM recovery: models unloaded, VRAM freed")
                    return True
                body = await response.text()
                logger.warning(
                    f"OOM recovery returned HTTP {response.status}: {body[:200]}"
                )
                return False
    except asyncio.TimeoutError:
        logger.warning("OOM recovery timed out")
        return False
    except Exception as e:
        logger.error(f"OOM recovery error: {e}")
        return False
