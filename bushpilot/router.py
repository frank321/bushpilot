"""Lightweight stdlib-only HTTP/1.1 router for Bushpilot live model hot-swap.

Listens on the public server port ($MLX_PORT, default 38472) and proxies traffic to
the active MLX backend (mlx_vlm.server or mlx-dspark serve) on $MLX_BACKEND_PORT
(default 38473).

Responsibilities:
1. Serves GET /v1/models and GET /models directly from ~/.bushpilot/state/mlx-roster.json
   so coding tools (Cline, OpenCode, Pi, etc.) see the exact curated session roster
   instead of raw Hugging Face cache directories or a single static model.
2. Inspects POST /v1/chat/completions, /v1/messages, /v1/responses, and /v1/completions:
   - If the request's `model` field matches a DIFFERENT model in mlx-roster.json than
     the currently active model in mlx-loaded.txt, coordinates an in-place backend
     hot-swap with start_mlx_server.sh (via mlx-swap-request.txt) while holding the
     client HTTP connection open.
   - Rewrites `model` in the forwarded request body to the active model's Hugging Face
     repo ID so that short Bushpilot catalog IDs (e.g. `qwen3.5-9b-mlx-8bit`) and
     Claude Code background probe aliases (`bushpilot-active`, `claude-haiku-*`) never
     404 or trigger an unmanaged in-process reload inside mlx_vlm.server.
3. Streams SSE and chunked/content-length responses back to the caller with zero
   buffering overhead.
"""
import argparse
import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

INFERENCE_PATHS = {
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/messages",
    "/messages",
    "/v1/responses",
    "/responses",
    "/v1/completions",
    "/completions",
}

MODELS_PATHS = {"/v1/models", "/models"}

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def read_loaded_state(state_file):
    """Parse mlx-loaded.txt: `hf_id|context_window|kv_bits|local_id|backend`."""
    try:
        with open(state_file) as f:
            raw = f.read().strip()
    except OSError:
        return None
    if not raw:
        return None
    parts = raw.split("|")
    return {
        "huggingface": parts[0] if len(parts) > 0 else "",
        "context_window": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
        "kv_bits": parts[2] if len(parts) > 2 else "",
        "id": parts[3] if len(parts) > 3 else "",
        "backend": parts[4] if len(parts) > 4 else "mlx_vlm",
        "raw": raw,
    }


def read_roster(roster_file):
    """Read mlx-roster.json (`{"models": [...]}`). Returns a list of model dicts."""
    try:
        with open(roster_file) as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("models"), list):
            return data["models"]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def strip_provider_prefix(model_str):
    """Strip common tool provider prefixes (`mlx-direct/`, `openai/`) when matching."""
    if not isinstance(model_str, str):
        return ""
    for prefix in ("mlx-direct/", "openai/"):
        if model_str.startswith(prefix):
            return model_str[len(prefix):]
    return model_str


def is_background_alias(model_str):
    """True when `model_str` is a non-switching alias (e.g. Claude Code's background
    Haiku probe `bushpilot-active` or `claude-haiku-*`) that should route to whatever
    model is currently loaded without triggering a hot-swap."""
    cleaned = strip_provider_prefix(model_str).strip()
    if not cleaned:
        return True
    if cleaned == "bushpilot-active":
        return True
    if cleaned.startswith("claude-haiku-"):
        return True
    return False


def match_roster_entry(requested_model, roster, loaded_state):
    """Resolve `requested_model` against `roster` and `loaded_state`.

    Returns `(target_entry, needs_swap)`:
    - `(None, False)` if `requested_model` already matches the currently loaded model,
      is a background alias, or is not in `roster`.
    - `(entry, True)` if `requested_model` matches a roster entry that differs from the
      currently loaded model.
    """
    if is_background_alias(requested_model):
        return None, False

    cleaned = strip_provider_prefix(requested_model).strip()
    loaded_id = (loaded_state or {}).get("id") or ""
    loaded_hf = (loaded_state or {}).get("huggingface") or ""

    # If the request explicitly names the currently loaded catalog ID, no swap needed.
    if loaded_id and cleaned == loaded_id:
        return None, False

    # Look for an exact catalog `id` match in the roster first (most specific, since two
    # catalog entries can share one `huggingface` repo across different backends/drafters).
    for entry in roster:
        if entry.get("id") == cleaned:
            if loaded_id and entry.get("id") == loaded_id:
                return None, False
            return entry, True

    # If it matches the currently loaded Hugging Face repo ID, keep the current model.
    if loaded_hf and cleaned == loaded_hf:
        return None, False

    # Otherwise match by Hugging Face repo ID in the roster.
    for entry in roster:
        if entry.get("huggingface") == cleaned:
            if loaded_id and entry.get("id") == loaded_id:
                return None, False
            return entry, True

    return None, False


def build_openai_models_response(roster, loaded_state):
    """Build an OpenAI-compatible `GET /v1/models` payload from the active roster."""
    created = int(time.time())
    data = []
    seen = set()

    for entry in roster:
        hf_id = entry.get("huggingface")
        cat_id = entry.get("id")
        # Expose the primary identifier (huggingface repo ID, plus catalog ID if distinct
        # and requested). Putting huggingface first and catalog ID when multiple variants
        # share a repo keeps `/v1/models` clean and unambiguous.
        model_key = cat_id if entry.get("expose_catalog_id") else (hf_id or cat_id)
        if not model_key or model_key in seen:
            continue
        seen.add(model_key)
        data.append({
            "id": model_key,
            "object": "model",
            "created": created,
            "owned_by": "bushpilot",
            "bushpilot_id": cat_id,
            "huggingface": hf_id,
            "context_window": entry.get("recommended_context"),
            "active": bool(
                loaded_state
                and (
                    (cat_id and loaded_state.get("id") == cat_id)
                    or (not loaded_state.get("id") and loaded_state.get("huggingface") == hf_id)
                )
            ),
        })

    if not data and loaded_state and loaded_state.get("huggingface"):
        data.append({
            "id": loaded_state["huggingface"],
            "object": "model",
            "created": created,
            "owned_by": "bushpilot",
            "bushpilot_id": loaded_state.get("id") or loaded_state["huggingface"],
            "huggingface": loaded_state["huggingface"],
            "context_window": loaded_state.get("context_window"),
            "active": True,
        })

    return {"object": "list", "data": data}


class RouterConfig:
    def __init__(self, backend_host, backend_port, state_dir, swap_timeout=300.0):
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.state_dir = state_dir
        self.state_file = os.path.join(state_dir, "mlx-loaded.txt")
        self.roster_file = os.path.join(state_dir, "mlx-roster.json")
        self.swap_request_file = os.path.join(state_dir, "mlx-swap-request.txt")
        self.swap_error_file = os.path.join(state_dir, "mlx-swap-error.txt")
        self.swap_timeout = swap_timeout
        self.swap_lock = threading.Lock()

    def request_swap(self, target_id):
        """Coordinate a synchronous model hot-swap with start_mlx_server.sh.

        Writes `target_id` to `mlx-swap-request.txt` and waits until `mlx-loaded.txt`
        reflects `target_id` (and `mlx-swap-request.txt` is cleared) or `mlx-swap-error.txt`
        is written by the supervisor. Thread-safe: concurrent requests for a swap wait on
        `self.swap_lock`.
        """
        with self.swap_lock:
            loaded = read_loaded_state(self.state_file)
            if loaded and loaded.get("id") == target_id:
                return True, None

            try:
                if os.path.exists(self.swap_error_file):
                    os.remove(self.swap_error_file)
            except OSError:
                pass

            tmp_req = f"{self.swap_request_file}.tmp.{os.getpid()}.{threading.get_ident()}"
            try:
                with open(tmp_req, "w") as f:
                    f.write(f"{target_id}\n")
                os.replace(tmp_req, self.swap_request_file)
            except OSError as e:
                return False, f"Failed to write swap request: {e}"

            deadline = time.monotonic() + self.swap_timeout
            while time.monotonic() < deadline:
                if os.path.exists(self.swap_error_file):
                    try:
                        with open(self.swap_error_file) as f:
                            err_msg = f.read().strip()
                        os.remove(self.swap_error_file)
                    except OSError:
                        err_msg = "Model swap failed"
                    return False, err_msg or "Model swap failed"

                loaded = read_loaded_state(self.state_file)
                req_cleared = not os.path.exists(self.swap_request_file)
                if req_cleared and loaded and loaded.get("id") == target_id:
                    return True, None

                time.sleep(0.25)

            return False, f"Timed out waiting {int(self.swap_timeout)}s for server to switch to '{target_id}'"


class BushpilotRouterServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, router_config):
        super().__init__(server_address, RequestHandlerClass)
        self.router_config = router_config


class BushpilotRouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Suppress per-request access logs (/metrics and /health poll every 5s).
        pass

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _read_body(self):
        transfer_encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in transfer_encoding:
            chunks = []
            while True:
                line = self.rfile.readline()
                if not line:
                    break
                size_str = line.split(b";", 1)[0].strip()
                try:
                    size = int(size_str, 16)
                except ValueError:
                    break
                if size == 0:
                    # Consume trailing CRLF after 0-length chunk
                    self.rfile.readline()
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)  # trailing CRLF
            return b"".join(chunks)

        length_str = self.headers.get("Content-Length")
        if not length_str:
            return b""
        try:
            length = int(length_str)
        except ValueError:
            return b""
        return self.rfile.read(length) if length > 0 else b""

    def _handle_request(self, method):
        cfg = self.server.router_config
        path_only = self.path.split("?", 1)[0]

        if method == "GET" and path_only in MODELS_PATHS:
            roster = read_roster(cfg.roster_file)
            loaded = read_loaded_state(cfg.state_file)
            self._send_json(200, build_openai_models_response(roster, loaded))
            return

        body = self._read_body() if method in ("POST", "PUT", "PATCH") else b""

        if method == "POST" and path_only in INFERENCE_PATHS and body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None

            if isinstance(payload, dict):
                req_model = payload.get("model")
                roster = read_roster(cfg.roster_file)
                loaded = read_loaded_state(cfg.state_file)
                target_entry, needs_swap = match_roster_entry(req_model, roster, loaded)

                if needs_swap and target_entry:
                    ok, err = cfg.request_swap(target_entry["id"])
                    if not ok:
                        self._send_json(503, {
                            "error": {
                                "message": f"Bushpilot model hot-swap to '{target_entry['id']}' failed: {err}",
                                "type": "server_error",
                            },
                        })
                        return
                    loaded = read_loaded_state(cfg.state_file)

                # Always normalize `model` in the forwarded body to the active model's
                # Hugging Face repo ID so mlx_vlm.server never attempts its own uncalibrated
                # in-process reload or 404s on a catalog ID / Claude Haiku probe alias.
                if loaded and loaded.get("huggingface"):
                    if payload.get("model") != loaded["huggingface"]:
                        payload["model"] = loaded["huggingface"]
                        body = json.dumps(payload).encode("utf-8")

        self._proxy_to_backend(method, body)

    def _proxy_to_backend(self, method, body):
        cfg = self.server.router_config
        conn = http.client.HTTPConnection(cfg.backend_host, cfg.backend_port, timeout=600)
        try:
            forward_headers = {}
            for k, v in self.headers.items():
                if k.lower() in HOP_BY_HOP_HEADERS or k.lower() == "content-length":
                    continue
                forward_headers[k] = v
            if body is not None and method in ("POST", "PUT", "PATCH"):
                forward_headers["Content-Length"] = str(len(body))
            forward_headers["Connection"] = "close"

            conn.request(method, self.path, body=body if body else None, headers=forward_headers)
            resp = conn.getresponse()

            self.send_response(resp.status, resp.reason)
            is_chunked = False
            content_length = None
            for k, v in resp.getheaders():
                kl = k.lower()
                if kl in HOP_BY_HOP_HEADERS:
                    if kl == "transfer-encoding" and "chunked" in v.lower():
                        is_chunked = True
                    continue
                if kl == "content-length":
                    content_length = v
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()

            if method == "HEAD":
                return

            if is_chunked:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            elif content_length is not None:
                remaining = int(content_length)
                while remaining > 0:
                    chunk = resp.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    remaining -= len(chunk)
            else:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected mid-stream (e.g. user pressed Esc in coding tool).
            pass
        except (OSError, http.client.HTTPException) as e:
            try:
                self._send_json(502, {
                    "error": {
                        "message": f"Backend on {cfg.backend_host}:{cfg.backend_port} unavailable: {e}",
                        "type": "bad_gateway",
                    },
                })
            except OSError:
                pass
        finally:
            conn.close()

    def do_GET(self):
        self._handle_request("GET")

    def do_POST(self):
        self._handle_request("POST")

    def do_DELETE(self):
        self._handle_request("DELETE")

    def do_PUT(self):
        self._handle_request("PUT")

    def do_PATCH(self):
        self._handle_request("PATCH")

    def do_OPTIONS(self):
        self._handle_request("OPTIONS")

    def do_HEAD(self):
        self._handle_request("HEAD")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bushpilot live model hot-swap HTTP router")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=38472)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=38473)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--swap-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    cfg = RouterConfig(
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        state_dir=args.state_dir,
        swap_timeout=args.swap_timeout,
    )
    server = BushpilotRouterServer((args.host, args.port), BushpilotRouterHandler, cfg)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main(sys.argv[1:])
