import http.client
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bushpilot.cli import build_roster_entries
from bushpilot.hardware import _get_reclaimable_server_rss_gb
from bushpilot.router import (
    BushpilotRouterHandler,
    BushpilotRouterServer,
    RouterConfig,
    build_openai_models_response,
    is_background_alias,
    match_roster_entry,
    read_loaded_state,
    read_roster,
)


def _sample_roster():
    return [
        {
            "id": "gemma-4-26b-a4b-it-4bit",
            "huggingface": "mlx-community/gemma-4-26b-a4b-it-4bit",
            "backend": "mlx_vlm",
            "recommended_context": 262144,
            "expose_catalog_id": False,
        },
        {
            "id": "qwen3.5-9b-mlx-8bit",
            "huggingface": "mlx-community/Qwen3.5-9B-MLX-8bit",
            "backend": "mlx_vlm",
            "recommended_context": 131072,
            "expose_catalog_id": False,
        },
        {
            "id": "qwen3.5-9b-mlx-8bit-dflash",
            "huggingface": "mlx-community/Qwen3.5-9B-MLX-8bit",
            "backend": "mlx_dspark",
            "drafter": "z-lab/Qwen3.5-9B-DFlash",
            "recommended_context": 65536,
            "expose_catalog_id": True,
        },
    ]


def test_is_background_alias():
    assert is_background_alias("")
    assert is_background_alias("bushpilot-active")
    assert is_background_alias("mlx-direct/bushpilot-active")
    assert is_background_alias("claude-haiku-4-5-20251001")
    assert not is_background_alias("mlx-community/gemma-4-26b-a4b-it-4bit")


def test_match_roster_entry_distinguishes_active_vs_swap_vs_alias():
    roster = _sample_roster()
    loaded = {
        "huggingface": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "id": "gemma-4-26b-a4b-it-4bit",
        "backend": "mlx_vlm",
    }

    # Same model by HF id or catalog id -> no swap
    assert match_roster_entry("mlx-community/gemma-4-26b-a4b-it-4bit", roster, loaded) == (None, False)
    assert match_roster_entry("openai/mlx-community/gemma-4-26b-a4b-it-4bit", roster, loaded) == (None, False)
    assert match_roster_entry("gemma-4-26b-a4b-it-4bit", roster, loaded) == (None, False)

    # Background Haiku probe -> no swap
    assert match_roster_entry("bushpilot-active", roster, loaded) == (None, False)
    assert match_roster_entry("claude-haiku-4-5-20251001", roster, loaded) == (None, False)

    # Switching to another model by HF id or catalog id -> triggers swap
    target, needs_swap = match_roster_entry("mlx-community/Qwen3.5-9B-MLX-8bit", roster, loaded)
    assert needs_swap is True
    assert target["id"] == "qwen3.5-9b-mlx-8bit"

    target_bp, needs_swap_bp = match_roster_entry("[BushPilot] mlx-community/Qwen3.5-9B-MLX-8bit", roster, loaded)
    assert needs_swap_bp is True
    assert target_bp["id"] == "qwen3.5-9b-mlx-8bit"

    target_dflash, needs_swap_dflash = match_roster_entry("qwen3.5-9b-mlx-8bit-dflash", roster, loaded)
    assert needs_swap_dflash is True
    assert target_dflash["id"] == "qwen3.5-9b-mlx-8bit-dflash"

    # Unknown model not in roster -> pass through to active without swap
    assert match_roster_entry("some-unknown-model", roster, loaded) == (None, False)


def test_build_openai_models_response_marks_active_and_exposes_catalog_id():
    roster = _sample_roster()
    loaded = {
        "huggingface": "mlx-community/Qwen3.5-9B-MLX-8bit",
        "id": "qwen3.5-9b-mlx-8bit-dflash",
        "backend": "mlx_dspark",
    }
    resp = build_openai_models_response(roster, loaded)
    assert resp["object"] == "list"
    ids = [m["id"] for m in resp["data"]]
    assert ids == [
        "mlx-community/gemma-4-26b-a4b-it-4bit",
        "mlx-community/Qwen3.5-9B-MLX-8bit",
        "qwen3.5-9b-mlx-8bit-dflash",
    ]
    active_flags = {m["id"]: m["active"] for m in resp["data"]}
    assert active_flags == {
        "mlx-community/gemma-4-26b-a4b-it-4bit": False,
        "mlx-community/Qwen3.5-9B-MLX-8bit": False,
        "qwen3.5-9b-mlx-8bit-dflash": True,
    }


def test_build_roster_entries_auto_populates_cached_and_respects_multi_select(monkeypatch):
    def _make_row(mid, hf_id, backend="mlx_vlm", verdict="GREAT", fits=True, released="2026-04-01"):
        return {
            "model": {
                "id": mid,
                "huggingface": hf_id,
                "size_gb": 10.0,
                "released": released,
                "serving": {"backend": backend},
                "tool_calling": {"verdict": verdict},
                "modalities": ["text", "image"],
            },
            "fits": fits,
            "recommended_context": 131072 if fits else 0,
            "kv_bits": None,
            "kv_bits_apc_caveat": False,
            "apc_num_blocks": None,
            "estimated_tok_s": 45.0,
            "concurrency": 1,
            "prefill_step_size": 4096,
        }

    rows = [
        _make_row("model-a", "org/model-a", released="2026-04-04"),
        _make_row("model-b", "org/model-b", verdict="OK", released="2026-04-03"),
        _make_row("model-b-dflash", "org/model-b", backend="mlx_dspark", released="2026-04-02"),
        _make_row("model-c-uncached", "org/model-c", released="2026-04-01"),
        _make_row("model-d-unsupported", "org/model-d", verdict="NOT_SUPPORTED", released="2026-04-05"),
    ]

    monkeypatch.setattr(
        "bushpilot.cli.is_model_cached",
        lambda m: m["huggingface"] in ("org/model-a", "org/model-b"),
    )

    # Single selection: auto-populates cached fitting models (model-b), skips duplicate HF
    # repo in auto-mode, skips uncached and NOT_SUPPORTED.
    auto_roster = build_roster_entries(rows, ["model-a"], max_models=5)
    assert [r["id"] for r in auto_roster] == ["model-a", "model-b"]
    assert auto_roster[0]["expose_catalog_id"] is False
    assert auto_roster[1]["expose_catalog_id"] is False

    # Explicit multi-selection (2+) with shared HF repo: uses exact user selection and sets
    # expose_catalog_id=True on entries sharing a Hugging Face repo ID.
    explicit_roster = build_roster_entries(rows, ["model-a", "model-b", "model-b-dflash"], max_models=5)
    assert [r["id"] for r in explicit_roster] == ["model-a", "model-b", "model-b-dflash"]
    assert explicit_roster[0]["expose_catalog_id"] is False
    assert explicit_roster[1]["expose_catalog_id"] is True
    assert explicit_roster[2]["expose_catalog_id"] is True


def test_reclaimable_server_rss_env_var(monkeypatch):
    monkeypatch.delenv("BUSHPILOT_RECLAIMABLE_RSS_GB", raising=False)
    assert _get_reclaimable_server_rss_gb() == 0.0

    monkeypatch.setenv("BUSHPILOT_RECLAIMABLE_RSS_GB", "14.5")
    assert _get_reclaimable_server_rss_gb() == 14.5

    monkeypatch.setenv("BUSHPILOT_RECLAIMABLE_RSS_GB", "invalid")
    assert _get_reclaimable_server_rss_gb() == 0.0


def test_router_end_to_end_models_forwarding_and_hot_swap(tmp_path):
    # Start a mock MLX backend server that echoes back what model it received
    received_bodies = []

    class MockBackendHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
            received_bodies.append(body)
            resp = json.dumps({"id": "chatcmpl-test", "received_model": body.get("model")}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    backend_srv = ThreadingHTTPServer(("127.0.0.1", 0), MockBackendHandler)
    backend_port = backend_srv.server_address[1]
    backend_thread = threading.Thread(target=backend_srv.serve_forever, daemon=True)
    backend_thread.start()

    state_dir = str(tmp_path)
    state_file = os.path.join(state_dir, "mlx-loaded.txt")
    roster_file = os.path.join(state_dir, "mlx-roster.json")
    swap_req_file = os.path.join(state_dir, "mlx-swap-request.txt")

    with open(state_file, "w") as f:
        f.write("mlx-community/gemma-4-26b-a4b-it-4bit|262144|8|gemma-4-26b-a4b-it-4bit|mlx_vlm\n")
    with open(roster_file, "w") as f:
        json.dump({"models": _sample_roster()}, f)

    cfg = RouterConfig("127.0.0.1", backend_port, state_dir, swap_timeout=5.0)
    router_srv = BushpilotRouterServer(("127.0.0.1", 0), BushpilotRouterHandler, cfg)
    router_port = router_srv.server_address[1]
    router_thread = threading.Thread(target=router_srv.serve_forever, daemon=True)
    router_thread.start()

    try:
        # 1. GET /v1/models returns the curated roster
        conn = http.client.HTTPConnection("127.0.0.1", router_port, timeout=5)
        conn.request("GET", "/v1/models")
        r = conn.getresponse()
        assert r.status == 200
        models_payload = json.loads(r.read().decode("utf-8"))
        conn.close()
        assert len(models_payload["data"]) == 3
        assert len(read_roster(roster_file)) == 3
        assert read_loaded_state(state_file)["id"] == "gemma-4-26b-a4b-it-4bit"

        # 2. POST /v1/messages with background Haiku alias rewrites model to active HF id without swapping
        conn = http.client.HTTPConnection("127.0.0.1", router_port, timeout=5)
        conn.request(
            "POST",
            "/v1/messages",
            body=json.dumps({"model": "bushpilot-active", "messages": []}),
            headers={"Content-Type": "application/json"},
        )
        r = conn.getresponse()
        assert r.status == 200
        resp_data = json.loads(r.read().decode("utf-8"))
        conn.close()
        assert resp_data["received_model"] == "mlx-community/gemma-4-26b-a4b-it-4bit"
        assert not os.path.exists(swap_req_file)

        # 3. POST /v1/chat/completions requesting a different roster model triggers swap handshake
        def mock_supervisor():
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                if os.path.exists(swap_req_file):
                    with open(swap_req_file) as rf:
                        target_id = rf.read().strip()
                    if target_id == "qwen3.5-9b-mlx-8bit":
                        with open(state_file, "w") as sf:
                            sf.write("mlx-community/Qwen3.5-9B-MLX-8bit|131072|none|qwen3.5-9b-mlx-8bit|mlx_vlm\n")
                        os.remove(swap_req_file)
                        return
                time.sleep(0.05)

        sup_thread = threading.Thread(target=mock_supervisor, daemon=True)
        sup_thread.start()

        conn = http.client.HTTPConnection("127.0.0.1", router_port, timeout=5)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps({"model": "mlx-community/Qwen3.5-9B-MLX-8bit", "messages": []}),
            headers={"Content-Type": "application/json"},
        )
        r = conn.getresponse()
        assert r.status == 200
        resp_data = json.loads(r.read().decode("utf-8"))
        conn.close()
        sup_thread.join(timeout=2)
        assert resp_data["received_model"] == "mlx-community/Qwen3.5-9B-MLX-8bit"
    finally:
        router_srv.shutdown()
        router_srv.server_close()
        backend_srv.shutdown()
        backend_srv.server_close()
