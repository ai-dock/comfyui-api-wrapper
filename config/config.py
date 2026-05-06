import os
from pathlib import Path
from urllib.parse import urljoin

# Load .env file if it exists (before reading environment variables)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass

# Base API configuration.
#
# Multi-backend support. The wrapper can fan out across N ComfyUI
# processes for parallelism — typically 1 per GPU on a multi-GPU
# host, occasionally 2+ per GPU when the model fits multiple times
# in VRAM. Set `COMFYUI_BACKENDS` to a comma-separated list of base
# URLs:
#
#     COMFYUI_BACKENDS="http://127.0.0.1:8188,http://127.0.0.1:8189"
#
# When unset, falls back to single-backend behaviour using
# `COMFYUI_API_BASE` (default `http://127.0.0.1:8188`). Existing
# deployments need no changes.
COMFYUI_API_BASE = os.getenv('COMFYUI_API_BASE', 'http://127.0.0.1:8188')


def _parse_backends(raw, fallback: str) -> list:
    """Comma-separated URL list → cleaned [base_url, ...]; fallback
    when empty. Trailing slashes stripped so urljoin behaves the
    same regardless of how the operator wrote the URL."""
    if not raw:
        return [fallback.rstrip('/')]
    out = []
    for part in raw.split(','):
        url = part.strip().rstrip('/')
        if url and url not in out:
            out.append(url)
    return out or [fallback.rstrip('/')]


COMFYUI_BACKENDS = _parse_backends(os.getenv('COMFYUI_BACKENDS'), COMFYUI_API_BASE)


def comfyui_urls(base: str) -> dict:
    """Build the full set of ComfyUI endpoint URLs for a given
    backend's HTTP base. Used by GenerationWorker per-instance and
    by /health to walk every backend.

    `prompt`, `queue`, `history`, `interrupt`, `free`, `system_stats`
    are HTTP; `websocket` is ws://-prefixed.
    """
    base = base.rstrip('/')
    return {
        "base":         base,
        "prompt":       urljoin(base + '/', 'prompt'),
        "queue":        urljoin(base + '/', 'queue'),
        "history":      urljoin(base + '/', 'history'),
        "interrupt":    urljoin(base + '/', 'api/interrupt'),
        "free":         urljoin(base + '/', 'free'),
        "system_stats": urljoin(base + '/', 'system_stats'),
        "websocket":    base.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws',
    }


# Back-compat module-level constants — point at the FIRST backend.
# New code reads URLs through `comfyui_urls(backend)` so it works
# against any of N backends.
_FIRST_BACKEND = comfyui_urls(COMFYUI_BACKENDS[0])
COMFYUI_API_PROMPT       = _FIRST_BACKEND["prompt"]
COMFYUI_API_QUEUE        = _FIRST_BACKEND["queue"]
COMFYUI_API_HISTORY      = _FIRST_BACKEND["history"]
COMFYUI_API_INTERRUPT    = _FIRST_BACKEND["interrupt"]
COMFYUI_API_FREE         = _FIRST_BACKEND["free"]
COMFYUI_API_SYSTEM_STATS = _FIRST_BACKEND["system_stats"]
COMFYUI_API_WEBSOCKET    = _FIRST_BACKEND["websocket"]

# WebSocket timeouts. Tuneable via env so an operator can stretch
# them for unusually long workflows without a code change.
#   INITIAL: how long to wait for the first WebSocket message after
#            posting the workflow (large diffusion models' first
#            inference step can be slow if not already hot in VRAM).
#   MESSAGE: how long to wait between subsequent WS messages.
#   MAX_NO_MESSAGE_RETRIES: when a WS receive() times out, how many
#            times to fall back to polling the queue + history
#            before treating the job as gone.
#   MAX_RECONNECTS: when the WS closes mid-job and the job is still
#            running on ComfyUI's side, how many times to reconnect.
WEBSOCKET_INITIAL_TIMEOUT       = float(os.getenv("WEBSOCKET_INITIAL_TIMEOUT", "60"))
WEBSOCKET_MESSAGE_TIMEOUT       = float(os.getenv("WEBSOCKET_MESSAGE_TIMEOUT", "120"))
WEBSOCKET_MAX_NO_MESSAGE_RETRIES = int(os.getenv("WEBSOCKET_MAX_NO_MESSAGE_RETRIES", "3"))
WEBSOCKET_MAX_RECONNECTS        = int(os.getenv("WEBSOCKET_MAX_RECONNECTS", "10"))

# Cache configuration
CACHE_TYPE = "redis" if os.getenv("API_CACHE", "").lower() == "redis" else "memory"
CACHE_TTL = int(os.getenv("API_CACHE_TTL", 21600))  # 6 hours as default

# Directory configuration using pathlib
COMFYUI_INSTALL_DIR = Path(os.getenv('COMFYUI_INSTALL_PATH', '/workspace/ComfyUI'))
INPUT_DIR = COMFYUI_INSTALL_DIR / 'input'
OUTPUT_DIR = COMFYUI_INSTALL_DIR / 'output'

# S3 Configuration (fallback from environment)
S3_CONFIG = {
    "access_key_id": os.getenv("S3_ACCESS_KEY_ID", ""),
    "secret_access_key": os.getenv("S3_SECRET_ACCESS_KEY", ""),
    "endpoint_url": os.getenv("S3_ENDPOINT_URL", ""),
    "bucket_name": os.getenv("S3_BUCKET_NAME", ""),
    "region": os.getenv("S3_REGION", ""),
    "connect_timeout": int(os.getenv("S3_CONNECT_TIMEOUT", "60")),
    "connect_attempts": int(os.getenv("S3_CONNECT_ATTEMPTS", "3"))
}

# Check if S3 is configured via environment
S3_ENABLED = bool(
    S3_CONFIG["access_key_id"] and 
    S3_CONFIG["secret_access_key"] and 
    S3_CONFIG["bucket_name"]
)

# Webhook Configuration (fallback from environment)
WEBHOOK_CONFIG = {
    "url": os.getenv("WEBHOOK_URL", ""),
    "timeout": int(os.getenv("WEBHOOK_TIMEOUT", "30"))
}

# Check if webhook is configured via environment
WEBHOOK_ENABLED = bool(WEBHOOK_CONFIG["url"])

# Worker Configuration
WORKER_CONFIG = {
    "preprocess_workers": int(os.getenv("PREPROCESS_WORKERS", "2")),
    "generation_workers": int(os.getenv("GENERATION_WORKERS", "1")),
    "postprocess_workers": int(os.getenv("POSTPROCESS_WORKERS", "2")),
    "max_queue_size": int(os.getenv("MAX_QUEUE_SIZE", "100"))
}

# Redis Configuration (if using Redis cache)
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "db": int(os.getenv("REDIS_DB", "0")),
    "password": os.getenv("REDIS_PASSWORD", ""),
    "decode_responses": True
}

# Development/Debug Configuration (actually used for debug output)
DEBUG_ENABLED = os.getenv("DEBUG", "false").lower() == "true"

# Print configuration summary if debug enabled
if DEBUG_ENABLED:
    print("🔧 Configuration Summary:")
    print(f"   ComfyUI API: {COMFYUI_API_BASE}")
    print(f"   Cache Type: {CACHE_TYPE}")
    print(f"   Workers: {WORKER_CONFIG['preprocess_workers']}/{WORKER_CONFIG['generation_workers']}/{WORKER_CONFIG['postprocess_workers']}")
    print(f"   S3 Enabled: {S3_ENABLED}")
    print(f"   Webhook Enabled: {WEBHOOK_ENABLED}")
    if os.path.exists('.env'):
        print("   📄 .env file loaded")
    else:
        print("   📄 No .env file found")