# ComfyUI API Wrapper

A FastAPI wrapper for ComfyUI that provides a scalable, production-ready interface for generating images using ComfyUI workflows. The wrapper handles queuing, preprocessing, generation coordination, and postprocessing with support for async, synchronous, and streaming responses.

**Source Code**: [AI-Dock GitHub](https://github.com/ai-dock/comfyui-api-wrapper)

## Features

- **Asynchronous request handling** with webhook notifications
- **Synchronous requests** that wait for completion  
- **Real-time streaming** of generation progress with queue positions
- **Automatic asset management** and S3 upload
- **Queue position tracking** and estimated wait times
- **Workflow modification system** for dynamic workflow processing
- **Redis or in-memory caching** support
- **Horizontal scaling** with configurable worker pools

## Quick Start

1. **Install dependencies:**
   ```bash
   apt-get update
   apt-get install libmagic1
   uv pip install -r requirements.txt
   ```

2. **Configure the service (optional):**
   ```bash
   # Copy example configuration
   cp .env.example .env
   
   # Edit .env file with your settings
   # Or set environment variables directly
   export COMFYUI_API_BASE=http://127.0.0.1:8188
   ```

3. **Run the service:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. **Access the API:**
   - Readme: /
   - Interactive docs: /docs
   - Health check: /health

## API Endpoints

### Generation Endpoints

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/generate` | POST | Async generation request | 202 with request ID |
| `/generate/sync` | POST | Synchronous generation | 200 with final result |
| `/generate/stream` | POST | Streaming progress updates | SSE stream |

### Utility Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/result/{request_id}` | GET | Get generation result |
| `/queue-info` | GET | Current queue status |
| `/health` | GET | Service health check |
| `/docs` | GET | Interactive API documentation |

## Request Payload

```json
{
  "input": {
    "request_id": "optional-uuid-v4",
    "modifier": "ModifierClassName",
    "modifications": {
      "parameter1": "value1",
      "parameter2": 42
    },
    "workflow_json": {
      // Alternative to modifier: direct ComfyUI workflow
    },
    "s3": {
      "access_key_id": "your-access-key",
      "secret_access_key": "your-secret-key", 
      "endpoint_url": "https://s3.amazonaws.com",
      "bucket_name": "your-bucket",
      "region": "us-east-1"
    },
    "webhook": {
      "url": "https://your-webhook-endpoint.com",
      "extra_params": {
        "custom_field": "value"
      }
    }
  }
}
```

## Usage Examples

### Synchronous Request

```python
import requests

payload = {
    "input": {
        "modifier": "YourModifier",
        "modifications": {"param": "value"}
    }
}

response = requests.post(
    "http://localhost:8000/generate/sync?timeout=300",
    json=payload
)
result = response.json()
print(f"Status: {result['status']}")
```

### Streaming Request

```python
import requests
import json
import sseclient

response = requests.post(
    "http://localhost:8000/generate/stream", 
    json=payload,
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    if event.data:
        data = json.loads(event.data)
        if data.get("status") == "final_result":
            print("Generation complete!")
            break
        else:
            pos = data.get("queue_position", {})
            if pos.get("position"):
                print(f"Position {pos['position']} of {pos['queue_size']} in {pos['current_queue']}")
```

### JavaScript Streaming

```javascript
const eventSource = new EventSource('/generate/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
});

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.status === 'final_result') {
        console.log('Complete:', data.result);
        eventSource.close();
    } else {
        console.log(`Status: ${data.status} - ${data.message}`);
    }
};
```

## Configuration

Configure the service using environment variables:

### API Server Configuration
<table>
    <tr><th>Variable</th><th>Default</th><th>Description</th></tr>
    <tr><td>API_HOST</td><td>0.0.0.0</td><td>Host to bind the API server</td></tr>
    <tr><td>API_PORT</td><td>8000</td><td>Port for the API server</td></tr>
    <tr><td>API_WORKERS</td><td>1</td><td>Number of uvicorn worker processes</td></tr>
    <tr><td>API_RELOAD</td><td>false</td><td>Enable auto-reload for development</td></tr>
    <tr><td>API_LOG_LEVEL</td><td>info</td><td>API server log level</td></tr>
</table>

### ComfyUI Connection
```bash
COMFYUI_API_BASE=http://127.0.0.1:8188  # ComfyUI API endpoint
```

### Worker Configuration
```bash
PREPROCESS_WORKERS=3          # Number of preprocessing workers
GENERATION_WORKERS=2          # Number of generation workers  
POSTPROCESS_WORKERS=3         # Number of postprocessing workers
MAX_QUEUE_SIZE=100           # Maximum queue size
```

### Cache Configuration
```bash
API_CACHE=redis              # 'redis' or 'memory'
REDIS_HOST=localhost         # Redis host (if using Redis)
REDIS_PORT=6379             # Redis port
REDIS_DB=0                  # Redis database
```

### S3 Configuration (Optional)
```bash
S3_ACCESS_KEY_ID=your-key
S3_SECRET_ACCESS_KEY=your-secret
S3_BUCKET_NAME=your-bucket
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_REGION=us-east-1
```

### Webhook Configuration (Optional)
```bash
WEBHOOK_URL=https://your-webhook.com  # Default webhook URL
WEBHOOK_TIMEOUT=30                   # Webhook timeout in seconds
```

## Workflow Modifiers

The system supports two approaches for workflow processing:

### 1. Static Workflow Modifiers

Create modifier classes bound to specific JSON workflow files:

```python
from modifiers.basemodifier import BaseModifier
import random

class Image2Image(BaseModifier):
    WORKFLOW_JSON = "workflows/image2image.json"
    
    def __init__(self, modifications={}):
        super().__init__(modifications)
    
    async def apply_modifications(self):
        # Modify specific nodes in the loaded workflow
        self.workflow["3"]["inputs"]["seed"] = await self.modify_workflow_value(
            "seed", random.randint(0, 2**32))
        self.workflow["6"]["inputs"]["text"] = await self.modify_workflow_value(
            "prompt", "")
        # URLs are automatically downloaded and replaced
        self.workflow["10"]["inputs"]["image"] = await self.modify_workflow_value(
            "https://example.com/image.jpg")
        
        # Call parent to handle URL downloads
        await super().apply_modifications()
```

### 2. Raw Workflow Processing

Send workflows directly in the payload using `workflow_json`. URLs anywhere in the workflow will be automatically downloaded and replaced with local file paths.

### Automatic URL Processing

The BaseModifier automatically:
- Scans workflows for URL strings
- Downloads assets to ComfyUI input directory  
- Uses MD5 hashing to cache downloads
- Replaces URLs with local filenames
- Detects MIME types for proper file extensions

### Example Requests

**Static Modifier:**
```json
{
  "input": {
    "modifier": "Image2Image",
    "modifications": {
      "prompt": "a beautiful sunset",
      "steps": 30,
      "input_image": "https://example.com/image.jpg"
    }
  }
}
```

**Raw Workflow:**
```json
{
  "input": {
    "workflow_json": {
      "10": {
        "inputs": {
          "image": "https://example.com/input.jpg",
          "upload": "image"
        },
        "class_type": "LoadImage"
      }
    }
  }
}
```

## Response Format

All endpoints return a standardized result object:

```json
{
  "id": "request-uuid",
  "status": "completed|failed|processing|generating|queued",
  "message": "Human-readable status message",
  "output": [
    {
      "filename": "output_001.png",
      "local_path": "/path/to/file",
      "url": "https://s3-url-if-uploaded.com/file.png",
      "type": "output",
      "node_id": "9",
      "output_type": "images"
    }
  ]
}
```

## Streaming Data Format

The `/generate/stream` endpoint provides real-time updates:

```json
{
  "request_id": "request-uuid",
  "status": "current-status",
  "message": "status-description", 
  "elapsed_time": 45.2,
  "queue_position": {
    "current_queue": "preprocessing|generation|postprocessing",
    "position": 3,
    "queue_size": 5,
    "estimated_wait_time": 120
  },
  "queue_info": {
    "preprocess_queue_size": 2,
    "generation_queue_size": 1, 
    "postprocess_queue_size": 0
  }
}
```

## Production Deployment

### Environment Setup

```bash
# Production with Redis cache
export COMFYUI_API_BASE=http://comfyui:8188
export API_CACHE=redis
export REDIS_HOST=redis
export PREPROCESS_WORKERS=5
export GENERATION_WORKERS=3
export POSTPROCESS_WORKERS=5

uvicorn main:app --host 0.0.0.0 --port 8000
```

### Scaling Considerations

- **Worker Pool Sizing**: Adjust worker counts based on ComfyUI capacity
- **Cache Strategy**: Use Redis for multi-instance deployments
- **Load Balancing**: Service supports horizontal scaling
- **Queue Monitoring**: Monitor `/queue-info` endpoint for capacity planning

## Architecture

```
Client Request → FastAPI → Preprocess Queue → Generation Queue → Postprocess Queue → Result
                     ↓           ↓                 ↓                ↓
                 Result Store  Workflow      ComfyUI API     Asset Upload
                              Modifier                         & Webhook
```

## Error Handling

The API uses standard HTTP status codes:

- **200 OK**: Request successful
- **202 Accepted**: Request queued for processing  
- **400 Bad Request**: Invalid payload
- **404 Not Found**: Request ID not found
- **500 Internal Server Error**: Processing failed

## Development

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Start ComfyUI**: Ensure ComfyUI is running on port 8188
4. **Run development server**: `uvicorn main:app --reload`


