"""Shared pytest fixtures for the model_advisor.py test suite.

model_advisor.py lives at the repo root as a standalone script, not inside a package,
so it isn't importable via pytest's normal rootdir-based sys.path insertion in every
invocation style (e.g. `pytest tests/` from a different cwd). Explicitly adding the repo
root here makes `import model_advisor` work regardless of how/where pytest is invoked
from, rather than relying on pytest's import-mode defaults.

Everything tested here is pure logic (string formatting, arithmetic, dict wrangling) --
no real Apple Silicon hardware, no MLX runtime, no models.json catalog required. See
verify_max_kv_size.sh / verify_long_context_memory.sh for the live-hardware checks that
intentionally stay out of this suite.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402 -- must follow the sys.path.insert above

import model_advisor as ma  # noqa: E402


@pytest.fixture
def dense_model():
    """A minimal, realistic dense-model entry -- Qwen3.5-9B-MLX-4bit-shaped, with a
    hybrid linear/full-attention kv_config (mirrors the real catalog entry's layer
    mix) so compute_kv_formula's layer_types branch gets exercised by default."""
    return {
        "id": "test-dense-4bit",
        "huggingface": "mlx-community/Test-Dense-4bit",
        "size_gb": 6.3,
        "type": "dense",
        "status": "tested",
        "modalities": ["text", "image", "video"],
        "released": "2026-02",
        "max_context_architectural": 262144,
        "active_weight_gb": 6.3,
        "kv_config": {
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_hidden_layers": 8,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention",
                "full_attention", "linear_attention", "linear_attention",
                "linear_attention", "full_attention",
            ],
        },
        "tool_calling": {"verdict": "SUPPORTED"},
    }


@pytest.fixture
def sliding_window_model():
    """gemma-4-shaped: a real sliding_window fixed-cost branch (compute_kv_formula),
    distinct from dense_model's linear-attention (~0 fixed cost) branch."""
    return {
        "id": "test-sliding-4bit",
        "huggingface": "mlx-community/Test-Sliding-4bit",
        "size_gb": 4.5,
        "type": "dense",
        "status": "tested",
        "modalities": ["text", "image"],
        "released": "2026-06",
        "max_context_architectural": 262144,
        "active_weight_gb": 4.5,
        "kv_config": {
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_hidden_layers": 8,
            "sliding_window": 512,
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "full_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention",
            ],
        },
        "tool_calling": {"verdict": "NOT_SUPPORTED"},
    }


@pytest.fixture
def moe_model_no_active_weight():
    """A MoE entry missing active_weight_gb -- estimate_speed_tok_s's documented
    "no reliable estimate" case."""
    return {
        "id": "test-moe-unknown-active",
        "huggingface": "mlx-community/Test-MoE-Unknown",
        "size_gb": 50.0,
        "type": "moe",
        "status": "untested",
        "modalities": ["text"],
        "released": "2026-07",
        "max_context_architectural": 131072,
        "kv_config": {
            "num_key_value_heads": 2,
            "head_dim": 128,
            "num_hidden_layers": 4,
            "layer_types": ["full_attention"] * 4,
        },
    }


@pytest.fixture
def sample_rows(dense_model, sliding_window_model, moe_model_no_active_weight):
    """Three build_recommendations() rows -- one that fits cleanly, one with a
    different modality/tool-support profile, and one that doesn't fit at all (a 50GB
    MoE model against 24GB free) -- reused across the picker-layout and sort tests so
    they exercise the exact row shapes build_recommendations actually produces, rather
    than a hand-rolled dict that could silently drift from the real one."""
    return ma.build_recommendations(
        [dense_model, sliding_window_model, moe_model_no_active_weight],
        free_wired_gb=24, chip_name="Apple M3 Max", gpu_cores=40,
    )


@pytest.fixture
def drafter_model():
    """A speculative-decoding entry (serving.drafter + measured_speedup), gemma-4-mtp-
    shaped, for estimate_speed_tok_s's multiplier branch and model_drafter_label."""
    return {
        "id": "test-drafter-mtp",
        "huggingface": "mlx-community/Test-Drafter-mtp",
        "size_gb": 4.5,
        "type": "dense",
        "status": "tested",
        "modalities": ["text", "image"],
        "released": "2026-06",
        "max_context_architectural": 262144,
        "active_weight_gb": 4.5,
        "kv_config": {
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_hidden_layers": 4,
            "layer_types": ["full_attention"] * 4,
        },
        "serving": {
            "backend": "mlx_vlm",
            "draft_model": "mlx-community/Test-Drafter-assistant-bf16",
            "draft_kind": "mtp",
            "drafter_size_gb": 0.16,
            "measured_speedup": 1.5,
            "measured_on": "Apple M5 Pro, 2026-09-29",
            "multimodal_capable": True,
        },
    }
