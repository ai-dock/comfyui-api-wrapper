import asyncio
import uuid
import logging
import json
from typing import Annotated, List
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Response, Body, Query, Request
from fastapi.responses import Response, StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from contextlib import asynccontextmanager
from anyio import create_task_group

from aiocache import Cache, SimpleMemoryCache
import time
import aiofiles

from config import CACHE_TYPE, WORKER_CONFIG, DEBUG_ENABLED
from requestmodels.models import Payload
from responses.result import Result
from workers.preprocess_worker import PreprocessWorker
from workers.generation_worker import GenerationWorker
from workers.postprocess_worker import PostprocessWorker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ComfyUI API Wrapper",
    description="FastAPI wrapper for ComfyUI with queue management",
    version="1.0.0",
    redirect_slashes=False  # Disable automatic slash redirects
)

# Add middleware to handle reverse proxy headers
@app.middleware("http")
async def add_reverse_proxy_headers(request: Request, call_next):
    """Handle reverse proxy headers to prevent redirect issues"""
    try:
        response = await call_next(request)
    except Exception as exc:
        # If the route raised (like 499 disconnect), just re-raise
        # or return a dummy JSONResponse to prevent noisy tracebacks.
        raise

    if "location" in response.headers:
        location = response.headers["location"]
        # If it's a redirect to the same path, remove it to prevent loops
        if location.endswith("//") or location == str(request.url):
            del response.headers["location"]

    return response


# Cache configuration - no changes needed, workers handle progress tracking
if CACHE_TYPE == "redis":
    request_store = Cache(Cache.REDIS, namespace="request_store")
    response_store = Cache(Cache.REDIS, namespace="response_store")
else:
    request_store = SimpleMemoryCache(namespace="request_store")
    response_store = SimpleMemoryCache(namespace="response_store")

# Processing queues (defined outside cache logic)
preprocess_queue = asyncio.Queue()    
generation_queue = asyncio.Queue()
postprocess_queue = asyncio.Queue()


@app.on_event("startup")
async def startup_event():
    """Initialize workers on startup"""
    try:
        asyncio.create_task(main())
        logger.info("Workers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize workers: {e}")
        raise


async def main():
    """Initialize and start all worker tasks"""
    worker_config = {
        "preprocess_queue": preprocess_queue,
        "generation_queue": generation_queue,
        "postprocess_queue": postprocess_queue,
        "request_store": request_store,
        "response_store": response_store,
    }

    # Create workers using configuration
    preprocess_workers = [PreprocessWorker(i, worker_config) for i in range(1, WORKER_CONFIG["preprocess_workers"] + 1)]
    preprocess_tasks = [asyncio.create_task(worker.work()) for worker in preprocess_workers]

    # Generation workers from configuration
    generation_workers = [GenerationWorker(i, worker_config) for i in range(1, WORKER_CONFIG["generation_workers"] + 1)]
    generation_tasks = [asyncio.create_task(worker.work()) for worker in generation_workers]

    postprocess_workers = [PostprocessWorker(i, worker_config) for i in range(1, WORKER_CONFIG["postprocess_workers"] + 1)]
    postprocess_tasks = [asyncio.create_task(worker.work()) for worker in postprocess_workers]

    logger.info(f"Started {len(preprocess_workers)} preprocess workers")
    logger.info(f"Started {len(generation_workers)} generation workers")
    logger.info(f"Started {len(postprocess_workers)} postprocess workers")

    # Wait indefinitely
    try:
        await asyncio.gather(*preprocess_tasks, *generation_tasks, *postprocess_tasks)
    except Exception as e:
        logger.error(f"Worker task failed: {e}")
        raise


# ===== DOCUMENTATION ENDPOINT =====
@app.get('/', response_class=HTMLResponse, include_in_schema=False)
async def documentation():
    """Serve the API documentation from README.md"""
    try:
        readme_path = Path("README.md")
        if not readme_path.exists():
            return """
            <html><body>
            <h1>ComfyUI API Wrapper</h1>
            <p>README.md not found. Please ensure README.md exists in the project root.</p>
            <p><a href="/docs">View Interactive API Documentation</a></p>
            </body></html>
            """
        
        async with aiofiles.open(readme_path, 'r', encoding='utf-8') as f:
            markdown_content = await f.read()
        
        # Convert markdown to HTML
        html_content = markdown_to_html(markdown_content)
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ComfyUI API Wrapper</title>
            <meta charset="utf-8">
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 1200px; 
                    margin: 0 auto; 
                    padding: 20px; 
                    line-height: 1.6; 
                    color: #333;
                }}
                h1, h2, h3 {{ color: #2c3e50; }}
                h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; margin-top: 30px; }}
                pre {{ 
                    background: #f8f9fa; 
                    padding: 15px; 
                    border-radius: 5px; 
                    overflow-x: auto;
                    border-left: 4px solid #3498db;
                }}
                code {{ 
                    background: #f1f2f6; 
                    padding: 2px 6px; 
                    border-radius: 3px;
                    font-family: 'Monaco', 'Consolas', monospace;
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 15px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #3498db; 
                    color: white;
                    font-weight: 600;
                }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                blockquote {{
                    border-left: 4px solid #3498db;
                    margin: 0;
                    padding: 10px 20px;
                    background-color: #f8f9fa;
                }}
                a {{ color: #3498db; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                ul, ol {{ margin: 10px 0; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
    except Exception as e:
        logger.error(f"Error serving documentation: {e}")
        return f"""
        <html><body>
        <h1>ComfyUI API Wrapper</h1>
        <p>Error loading documentation: {str(e)}</p>
        <p><a href="/docs">View Interactive API Documentation</a></p>
        </body></html>
        """


def markdown_to_html(markdown_text: str) -> str:
    """Simple markdown to HTML converter for basic formatting"""
    lines = markdown_text.split('\n')
    html_lines = []
    in_code_block = False
    in_table = False
    
    for line in lines:
        # Handle code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                html_lines.append('</pre>')
                in_code_block = False
            else:
                language = line.strip()[3:].strip()
                html_lines.append(f'<pre><code class="{language}">')
                in_code_block = True
            continue
        
        if in_code_block:
            html_lines.append(line)
            continue
        
        # Handle headers
        if line.startswith('#### '):
            html_lines.append(f'<h4>{line[5:]}</h4>')
        elif line.startswith('### '):
            html_lines.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('## '):
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('# '):
            html_lines.append(f'<h1>{line[2:]}</h1>')
        
        # Handle tables
        elif '|' in line and line.strip():
            if not in_table:
                html_lines.append('<table>')
                in_table = True
            
            if line.strip().startswith('|') and '---' in line:
                continue  # Skip separator line
            
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if cells:
                # Check if this is likely a header row (first table row)
                is_header = not any('</table>' in hl for hl in html_lines[-5:])
                tag = 'th' if is_header and len([l for l in html_lines if '<tr>' in l]) == 0 else 'td'
                
                row_html = '<tr>' + ''.join(f'<{tag}>{cell}</{tag}>' for cell in cells) + '</tr>'
                html_lines.append(row_html)
        
        # Handle lists
        elif line.strip().startswith('- '):
            if not html_lines or not html_lines[-1].startswith('<li>'):
                html_lines.append('<ul>')
            html_lines.append(f'<li>{line.strip()[2:]}</li>')
            
        elif line.strip().startswith('* '):
            if not html_lines or not html_lines[-1].startswith('<li>'):
                html_lines.append('<ul>')
            html_lines.append(f'<li>{line.strip()[2:]}</li>')
        
        # Handle paragraphs and empty lines
        elif line.strip() == '':
            if in_table:
                html_lines.append('</table>')
                in_table = False
            elif html_lines and html_lines[-1].startswith('<li>'):
                html_lines.append('</ul>')
            html_lines.append('<br>')
            
        else:
            # Regular paragraph
            if in_table:
                html_lines.append('</table>')
                in_table = False
            elif html_lines and html_lines[-1].startswith('<li>'):
                html_lines.append('</ul>')
            
            # Handle inline formatting
            formatted_line = line
            # Bold
            formatted_line = formatted_line.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
            # Italic  
            formatted_line = formatted_line.replace('*', '<em>', 1).replace('*', '</em>', 1)
            # Code
            formatted_line = formatted_line.replace('`', '<code>', 1).replace('`', '</code>', 1)
            
            html_lines.append(f'<p>{formatted_line}</p>')
    
    # Close any open elements
    if in_code_block:
        html_lines.append('</pre>')
    if in_table:
        html_lines.append('</table>')
    if html_lines and html_lines[-1].startswith('<li>'):
        html_lines.append('</ul>')
    
    return '\n'.join(html_lines)


# ===== ASYNC ENDPOINT (renamed from /payload) =====
@app.post('/generate', response_model=Result)
async def generate(
    response: Response,
    payload: Annotated[
        Payload,
        Body(
            openapi_examples=Payload.get_openapi_examples()
        ),
    ],
):
    """Submit a new generation request (async)"""
    if not payload.input.request_id:
        payload.input.request_id = str(uuid.uuid4())
    request_id = payload.input.request_id
    
    result_pending = Result(id=request_id)

    try:
        # Store request and initial result
        await request_store.set(request_id, payload)
        await response_store.set(request_id, result_pending)
        await preprocess_queue.put(request_id)
        
        logger.info(f"Queued request {request_id}")
        response.status_code = 202
        return result_pending
    except Exception as e:
        logger.error(f"Failed to queue request {request_id}: {e}")
        response.status_code = 500  # Internal Server Error
        failed_result = Result(
            id=request_id,
            status="failed",
            message=f"Failed to queue request: {str(e)}"
        )
        return failed_result

# ===== SYNCHRONOUS ENDPOINT =====
@asynccontextmanager
async def cancel_on_disconnect(request: Request, request_id: str):
    """Cancel work if the client disconnects prematurely."""
    async with create_task_group() as tg:
        async def watch_disconnect():
            try:
                while True:
                    message = await request.receive()
                    if message["type"] == "http.disconnect":
                        client = f'{request.client.host}:{request.client.port}' if request.client else '-:-'
                        logger.info(f'{client} - "{request.method} {request.url.path}" 499 DISCONNECTED for {request_id}')
                        await _mark_request_cancelled(request_id)
                        tg.cancel_scope.cancel()
                        break
            except asyncio.CancelledError:
                # Task cancelled by scope exit — swallow it
                pass

        tg.start_soon(watch_disconnect)
        try:
            yield
        finally:
            tg.cancel_scope.cancel()

@app.post("/generate/sync", response_model=Result, status_code=200)
async def generate_sync(
    request: Request,
    response: Response,
    payload: Annotated[
        Payload,
        Body(
            openapi_examples=Payload.get_openapi_examples()
        ),
    ],
):
    if not payload.input.request_id:
        payload.input.request_id = str(uuid.uuid4())
    request_id = payload.input.request_id

    result_pending = Result(id=request_id)
    await request_store.set(request_id, payload)
    await response_store.set(request_id, result_pending)
    await preprocess_queue.put(request_id)

    logger.info(f"Queued synchronous request {request_id}")

    try:
        async with cancel_on_disconnect(request, request_id):
            while True:
                result = await response_store.get(request_id)
                if result and result.status in ["completed", "failed", "timeout", "cancelled"]:
                    return result
                await asyncio.sleep(0.5)

    except asyncio.CancelledError:
        # Clean return instead of exception bubble
        return JSONResponse(
            status_code=499,
            content=Result(
                id=request_id,
                status="cancelled",
                message="Client closed connection"
            ).__dict__,
        )

# ===== STREAMING ENDPOINT =====
@app.post('/generate/stream')
async def generate_stream(
    payload: Annotated[
        Payload,
        Body(
            openapi_examples=Payload.get_openapi_examples()
        ),
    ],
):
    """Submit a request and stream status updates until completion"""
    if not payload.input.request_id:
        payload.input.request_id = str(uuid.uuid4())
    request_id = payload.input.request_id
    
    result_pending = Result(id=request_id)

    try:
        # Store request and initial result
        await request_store.set(request_id, payload)
        await response_store.set(request_id, result_pending)
        await preprocess_queue.put(request_id)
        
        logger.info(f"Starting stream for request {request_id}")
        
        # Return streaming response
        return StreamingResponse(
            _stream_status_updates(request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache", 
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start stream for request {request_id}: {e}")
        raise


# ===== HELPER FUNCTIONS =====

async def _mark_request_cancelled(request_id: str):
    """Helper to mark a request as cancelled in the response store"""
    try:
        result = await response_store.get(request_id)
        if result:
            # Only update if not already in a terminal state
            if result.status not in ['completed', 'failed', 'timeout', 'cancelled']:
                result.status = "cancelled"
                result.message = "Request cancelled due to client disconnection"
                await response_store.set(request_id, result)
                logger.info(f"Marked request {request_id} as cancelled")
            else:
                logger.debug(f"Request {request_id} already in terminal state: {result.status}")
        else:
            # Create a new cancelled result if none exists
            cancelled_result = Result(
                id=request_id,
                status="cancelled",
                message="Request cancelled due to client disconnection"
            )
            await response_store.set(request_id, cancelled_result)
            logger.info(f"Created cancelled result for request {request_id}")
            
    except Exception as e:
        logger.error(f"Failed to mark request {request_id} as cancelled: {e}")

async def _stream_status_updates(request_id: str):
    """Generator that yields Server-Sent Events for status updates using worker progress"""
    last_result = None
    last_queue_position = None
    poll_interval = 1.0  # Check for updates every second
    start_time = time.time()
    
    # Send initial event
    yield f"data: {json.dumps({'request_id': request_id, 'status': 'queued', 'message': 'Request queued', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
    
    while True:
        try:
            # Get current result from the store (updated by workers)
            current_result = await response_store.get(request_id)
            
            # Get current queue position
            queue_position = _get_queue_position(request_id)
            
            # Check if result has changed or queue position changed
            position_changed = queue_position != last_queue_position
            result_changed = current_result and (not last_result or _result_changed(last_result, current_result))
            
            if result_changed or position_changed:
                
                # Build event data from the result object
                event_data = {
                    "request_id": request_id,
                    "status": getattr(current_result, 'status', 'unknown') if current_result else 'queued',
                    "message": getattr(current_result, 'message', '') if current_result else 'Request queued',
                    "timestamp": datetime.utcnow().isoformat(),
                    "elapsed_time": round(time.time() - start_time, 1),
                    "queue_info": {
                        "preprocess_queue_size": preprocess_queue.qsize(),
                        "generation_queue_size": generation_queue.qsize(),
                        "postprocess_queue_size": postprocess_queue.qsize(),
                    },
                    "queue_position": queue_position
                }
                
                # Add output info if available
                if current_result and hasattr(current_result, 'output') and current_result.output:
                    event_data["output_count"] = len(current_result.output)
                
                yield f"data: {json.dumps(event_data)}\n\n"
                last_result = current_result
                last_queue_position = queue_position
            
            # Check if processing is complete
            if current_result and hasattr(current_result, 'status'):
                if current_result.status in ['completed', 'failed', 'timeout']:
                    # Send final result
                    final_data = {
                        "request_id": request_id,
                        "status": "final_result",
                        "result": _serialize_result(current_result),
                        "elapsed_time": round(time.time() - start_time, 1)
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"
                    break
            
            await asyncio.sleep(poll_interval)
            
        except Exception as e:
            error_data = {
                "request_id": request_id,
                "status": "error",
                "message": f"Stream error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            break


def _get_queue_position(request_id: str) -> dict:
    """Get the current position of request_id in queues"""
    position_info = {
        "current_queue": None,
        "position": 0,
        "queue_size": 0,
        "estimated_wait_time": None
    }
    
    try:
        # Check preprocess queue
        preprocess_items = list(preprocess_queue._queue)
        if request_id in preprocess_items:
            position = preprocess_items.index(request_id) + 1  # 1-based position
            position_info.update({
                "current_queue": "preprocessing",
                "position": position,
                "queue_size": len(preprocess_items),
                "estimated_wait_time": position * 30  # Rough estimate: 30s per preprocessing job
            })
            return position_info
        
        # Check generation queue
        generation_items = list(generation_queue._queue)
        if request_id in generation_items:
            position = generation_items.index(request_id) + 1
            position_info.update({
                "current_queue": "generation", 
                "position": position,
                "queue_size": len(generation_items),
                "estimated_wait_time": position * 120  # Rough estimate: 2min per generation job
            })
            return position_info
        
        # Check postprocess queue
        postprocess_items = list(postprocess_queue._queue)
        if request_id in postprocess_items:
            position = postprocess_items.index(request_id) + 1
            position_info.update({
                "current_queue": "postprocessing",
                "position": position, 
                "queue_size": len(postprocess_items),
                "estimated_wait_time": position * 20  # Rough estimate: 20s per postprocessing job
            })
            return position_info
        
        # Not in any queue - likely being processed or completed
        position_info.update({
            "current_queue": "processing",
            "position": 0,
            "queue_size": 0,
            "estimated_wait_time": 0
        })
        
    except Exception as e:
        logger.debug(f"Error getting queue position for {request_id}: {e}")
        position_info.update({
            "current_queue": "unknown",
            "position": 0,
            "queue_size": 0,
            "estimated_wait_time": None
        })
    
    return position_info


def _result_changed(old_result, new_result) -> bool:
    """Check if the result object has meaningfully changed"""
    if not old_result or not new_result:
        return True
    
    # Check key fields that indicate progress
    old_status = getattr(old_result, 'status', '')
    new_status = getattr(new_result, 'status', '')
    old_message = getattr(old_result, 'message', '')
    new_message = getattr(new_result, 'message', '')
    
    return old_status != new_status or old_message != new_message


def _serialize_result(result) -> dict:
    """Convert result object to dictionary for JSON serialization"""
    try:
        if hasattr(result, '__dict__'):
            # Get all attributes as dict
            result_dict = {}
            for key, value in result.__dict__.items():
                try:
                    # Test if value is JSON serializable
                    json.dumps(value)
                    result_dict[key] = value
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    result_dict[key] = str(value)
            return result_dict
        else:
            return {"data": str(result)}
    except Exception as e:
        return {"error": f"Serialization error: {str(e)}"}


@app.get('/result/{request_id}', response_model=Result, status_code=200)
async def result(request_id: str, response: Response):
    """Get the result of a processing request"""
    try:
        result = await response_store.get(request_id)
        if not result:
            result = Result(id=request_id, status="failed", message="Request ID not found")
            response.status_code = 404
        
        return result
    except Exception as e:
        logger.error(f"Failed to get result for {request_id}: {e}")
        result = Result(id=request_id, status="failed", message="Internal server error")
        response.status_code = 500
        return result

@app.post('/cancel/{request_id}', status_code=200)
async def cancel_request_simple(
    request_id: str,
    response: Response
):
    """Cancel a request by marking it as cancelled"""
    try:
        # Get the current result
        result = await response_store.get(request_id)
        
        if not result:
            response.status_code = 404
            return {"error": f"Request {request_id} not found"}
        
        # Check if already in a terminal state
        if result.status in ['completed', 'failed', 'timeout', 'cancelled']:
            return {
                "message": f"Request {request_id} is already {result.status}",
                "status": result.status
            }
        
        # Mark as cancelled
        result.status = "cancelled"
        result.message = "Request cancelled by client"
        await response_store.set(request_id, result)
        
        logger.info(f"Cancelled request {request_id}")
        
        return {
            "message": f"Successfully cancelled request {request_id}",
            "status": "cancelled"
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel request {request_id}: {e}")
        response.status_code = 500

@app.get('/queue-info', response_model=dict)
async def queue_info():
    """Get information about current queue sizes"""
    return {
        "preprocess_queue_size": preprocess_queue.qsize(),
        "generation_queue_size": generation_queue.qsize(),
        "postprocess_queue_size": postprocess_queue.qsize(),
    }


@app.get('/health', response_model=dict)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cache_type": CACHE_TYPE,
        "queues": {
            "preprocess": preprocess_queue.qsize(),
            "generation": generation_queue.qsize(), 
            "postprocess": postprocess_queue.qsize(),
        }
    }