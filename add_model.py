#!/usr/bin/env python3
"""
add_model.py — generate a models.json entry for a Hugging Face model.

Takes a repo id, a Hugging Face URL, or a plain model name:

  python3 add_model.py mlx-community/gemma-4-26b-a4b-it-8bit
  python3 add_model.py https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-8bit
  python3 add_model.py "gemma 4 26b a4b 8bit"      # searches HF, asks which result
  python3 add_model.py <any of the above> --write  # append to models.json (else print JSON)

Everything model_advisor.py needs to compute memory fit is read from the published repo
(config.json + safetensors sizes/headers): size, architectural context, kv_config, dense
vs MoE, input modalities, and — for MoE — the per-token active weight. The entry is
written as status "untested": nothing here proves the model runs under mlx_vlm or returns
tool calls. Run verify_max_kv_size.sh and a tool-calling check, write the display text,
then set status to "tested" by hand.

Set HF_TOKEN for gated repos. SSL_CERT_FILE is honored (TLS-intercepting proxies).
"""
import argparse
import json
import os
import re
import ssl
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request

HF = "https://huggingface.co"
MODELS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")
VERIFICATION_RESULTS_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "docs", "mlx_verification_results.json"
)
SWEEPS_DIR = os.path.expanduser("~/.bushpilot/calibration/sweeps")
REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
EXPERT_RE = re.compile(r"\.(experts|switch_mlp|switch_glu)\.")
NON_TEXT_PREFIXES = ("vision", "audio", "visual", "multi_modal_projector", "embed_vision", "embed_audio")


def log(msg):
    print(msg, file=sys.stderr)


def die(msg):
    log(f"error: {msg}")
    sys.exit(1)


def _ssl_context():
    cafile = os.environ.get("SSL_CERT_FILE")
    return ssl.create_default_context(cafile=cafile if cafile and os.path.isfile(cafile) else None)


def http_get(url, headers=None, timeout=30):
    req_headers = {"User-Agent": "bushpilot-add-model"}
    if os.environ.get("HF_TOKEN"):
        req_headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    req_headers.update(headers or {})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=req_headers),
                                    timeout=timeout, context=_ssl_context()) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        hint = " (gated or private repo? set HF_TOKEN)" if e.code in (401, 403) else ""
        die(f"HTTP {e.code} fetching {url}{hint}")
    except urllib.error.URLError as e:
        die(f"could not reach {url}: {e.reason}")


def http_json(url):
    return json.loads(http_get(url))


# ---------------------------------------------------------------------------
# Resolving the user's input to a repo id
# ---------------------------------------------------------------------------
def parse_url(text):
    parsed = urllib.parse.urlparse(text)
    if parsed.netloc not in ("huggingface.co", "www.huggingface.co", "hf.co"):
        return None
    parts = [p for p in parsed.path.split("/") if p]
    return "/".join(parts[:2]) if len(parts) >= 2 and REPO_RE.match("/".join(parts[:2])) else None


def search_repo(query, pick):
    url = f"{HF}/api/models?" + urllib.parse.urlencode(
        {"search": query, "filter": "mlx", "sort": "downloads", "direction": "-1", "limit": 10})
    results = http_json(url)
    results.sort(key=lambda r: not r["id"].startswith("mlx-community/"))  # stable: mlx-community first
    if not results:
        die(f"no MLX models on Hugging Face match '{query}'")
    if pick is None and not sys.stdin.isatty():
        pick = 1
    if pick is None:
        log(f"Matches for '{query}' (most downloaded first):")
        for i, r in enumerate(results, 1):
            log(f"  {i:2}) {r['id']}   ({r.get('downloads', 0):,} downloads)")
        try:
            answer = input(f"Pick [1-{len(results)}, default 1]: ").strip()
        except EOFError:
            answer = ""
        pick = int(answer) if answer.isdigit() else 1
    if not 1 <= pick <= len(results):
        die(f"--pick {pick} is out of range 1-{len(results)}")
    return results[pick - 1]["id"]


def resolve_repo(text, pick):
    text = text.strip()
    return parse_url(text) or (text if REPO_RE.match(text) and "://" not in text else None) \
        or search_repo(text, pick)


# ---------------------------------------------------------------------------
# Reading the repo
# ---------------------------------------------------------------------------
def safetensors_header(repo, filename):
    """Tensor name -> byte size, from just the file's JSON header (two range requests)."""
    url = f"{HF}/{repo}/resolve/main/{urllib.parse.quote(filename)}"
    (n,) = struct.unpack("<Q", http_get(url, {"Range": "bytes=0-7"}))
    header = json.loads(http_get(url, {"Range": f"bytes=8-{7 + n}"}))
    return {k: v["data_offsets"][1] - v["data_offsets"][0] for k, v in header.items() if k != "__metadata__"}


def active_weight_gb(repo, files, num_experts, top_k):
    """Bytes read per decoded token: every text-side tensor that isn't a routed expert,
    plus top_k/num_experts of the expert tensors. Vision/audio towers are excluded."""
    experts = other = 0
    for f in files:
        for name, size in safetensors_header(repo, f).items():
            if name.startswith(NON_TEXT_PREFIXES):
                continue
            if EXPERT_RE.search(name):
                experts += size
            else:
                other += size
    return round((other + experts * top_k / num_experts) / 2**30, 2), other, experts


def release_month(repo, info):
    """'YYYY-MM' the model came out: the base model's repo creation date when the repo
    names one (base_model tag, else the model card's base_model; the MLX conversion is
    usually later than the real release), else this repo's own creation date."""
    created = info.get("createdAt")
    card_base = (info.get("cardData") or {}).get("base_model")
    candidates = [t.split(":")[-1] for t in info.get("tags", []) if t.startswith("base_model:")]
    candidates += [card_base] if isinstance(card_base, str) else (card_base or [])
    if candidates and REPO_RE.match(candidates[0]) and candidates[0] != repo:
        try:
            created = json.loads(http_get(f"{HF}/api/models/{candidates[0]}")).get("createdAt") or created
        except SystemExit:
            pass  # base repo gone/gated: keep this repo's date
    return created[:7] if created else None


def extract_kv_config(t):
    """Extracts the kv_config dict consumed by bushpilot/kv.py from a HF text_config
    (or root config.json), including hybrid linear-attention (Qwen3.5/Qwen3-Next) and
    DeepSeek MLA geometry fields when present."""
    if t.get("num_hidden_layers") is None:
        die("config.json has no 'num_hidden_layers' — cannot compute the memory-fit numbers for this model")
    num_layers = t["num_hidden_layers"]
    heads = t.get("num_attention_heads")
    kv_heads = t.get("num_key_value_heads") or heads
    head_dim = t.get("head_dim") or (t["hidden_size"] // heads if heads and t.get("hidden_size") else None)
    if not (kv_heads and head_dim) and not t.get("kv_lora_rank"):
        die("config.json has no attention head geometry — cannot compute the KV-cache size")
    sliding = t.get("sliding_window")
    if t.get("use_sliding_window") is False:
        sliding = None
    kv_config = {"num_key_value_heads": kv_heads, "head_dim": head_dim, "num_hidden_layers": num_layers}
    for opt_key in (
        "global_head_dim",
        "num_global_key_value_heads",
        "num_kv_shared_layers",
        "layer_types",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "full_attention_interval",
        "kv_lora_rank",
        "qk_rope_head_dim",
    ):
        if t.get(opt_key) is not None:
            kv_config[opt_key] = t[opt_key]
    kv_config["sliding_window"] = sliding
    return kv_config


def build_entry(repo):
    info = http_json(f"{HF}/api/models/{repo}?blobs=true")
    files = [s for s in info.get("siblings", []) if s["rfilename"].endswith(".safetensors")]
    if not files:
        die(f"{repo} has no .safetensors files — not something mlx_vlm can load")
    config = json.loads(http_get(f"{HF}/{repo}/resolve/main/config.json"))
    t = config.get("text_config", config)

    kv_config = extract_kv_config(t)
    sliding = kv_config["sliding_window"]

    num_experts = t.get("num_experts") or t.get("num_local_experts") or t.get("n_routed_experts")
    top_k = t.get("top_k_experts") or t.get("num_experts_per_tok") or t.get("num_experts_per_token")
    is_moe = bool(num_experts and num_experts > 1)

    total_bytes = sum(s.get("size") or 0 for s in info["siblings"])  # whole repo, like the du-based size_gb elsewhere
    size_gb = round(total_bytes / 2**30, 2)
    max_ctx = t.get("max_position_embeddings") or config.get("max_position_embeddings")
    if not max_ctx:
        die("config.json has no max_position_embeddings")

    modalities = ["text"]
    if "vision_config" in config:
        modalities.append("image")
    if "video_token_id" in config or "video_token_id" in t:
        modalities.append("video")

    name = repo.split("/", 1)[1]
    entry = {
        "id": name.lower(),
        "huggingface": repo,
        "size_gb": size_gb,
        "architecture": config.get("model_type", "unknown"),
        "type": "moe" if is_moe else "dense",
        "status": "untested",
        "modalities": modalities,
        "released": release_month(repo, info),
    }
    if is_moe:
        if top_k:
            gb, other, experts = active_weight_gb(repo, [f["rfilename"] for f in files], num_experts, top_k)
            entry["active_weight_gb"] = gb
            entry["_active_weight_note"] = (
                f"Generated by add_model.py from the safetensors headers + config.json "
                f"(num_experts={num_experts}, top_k={top_k}): {other / 2**30:.2f}GB of always-read text tensors "
                f"+ {top_k}/{num_experts} of {experts / 2**30:.2f}GB expert tensors. Vision/audio towers excluded. "
                "Heuristic tensor-name matching — sanity-check against a measured decode speed.")
        else:
            entry["active_weight_gb"] = None
            entry["_active_weight_note"] = ("MoE, but config.json has no top-k expert count, so the active "
                                            "fraction is unknown; the picker shows 'speed unknown'.")
    entry["max_context_architectural"] = max_ctx
    sliding_note = f"; sliding_window={sliding} internally" if sliding else ""
    entry["display"] = {
        "tagline": f"{'MoE' if is_moe else 'Dense'} model, {size_gb:g} GB — untested",
        "good_for": "Not described yet. This entry was generated from the model's published config "
                    "and has not been run through this project's tests.",
        "usage_examples": [],
        "known_issues": "Untested: not yet run through verify_max_kv_size.sh or a live tool-calling check "
                        "on this project's mlx_vlm version. Memory fit is computed from the published config "
                        "and should be reliable; whether it serves correctly is unconfirmed.",
        "size_label": f"{size_gb:g} GB",
        "max_context_label": f"{max_ctx:,} tokens (architectural max){sliding_note}",
    }
    entry["tool_calling"] = {"verdict": "UNTESTED"}
    entry["kv_config"] = kv_config
    return entry


def write_entry(entry, path):
    with open(path) as f:
        catalog = json.load(f)
    for m in catalog["models"]:
        if m["id"] == entry["id"] or m["huggingface"] == entry["huggingface"]:
            die(f"{path} already has '{m['id']}' ({m['huggingface']}) — not adding a duplicate")
    catalog["models"].append(entry)
    with open(path, "w") as f:
        f.write(json.dumps(catalog, indent=4, ensure_ascii=False) + "\n")


def sync_verified_results(
    models_path=MODELS_JSON,
    verification_results_path=VERIFICATION_RESULTS_JSON,
    sweeps_dir=SWEEPS_DIR,
    write=False,
):
    """Ingests verification verdicts from `verification_results_path`
    (docs/mlx_verification_results.json) and empirical sweep results from `sweeps_dir`
    (~/.bushpilot/calibration/sweeps/*.sweep.json) into `models_path` (models.json).

    - Populates `verification` on any catalog entry whose HuggingFace repo id has
      `verdict == "SAFE"` in `mlx_verification_results.json` (when not already SAFE).
    - Populates `speed_correction_factor` from `derived_speed_correction_factor` in
      the newest matching `*.sweep.json`.
    - Populates `tool_calling` from `tool_calling_reverified` in the newest matching
      `*.sweep.json` when the catalog entry's `tool_calling.verdict` is unset or
      `"UNTESTED"`.
    - Promotes `status` from `"untested"` to `"tested"` ONLY when BOTH
      `verification.verdict == "SAFE"` AND `tool_calling.verdict` is in
      `("SUPPORTED", "NOT_SUPPORTED")`, preserving AGENTS.md Invariant #2."""
    with open(models_path) as f:
        catalog = json.load(f)

    ver_models = {}
    if verification_results_path and os.path.isfile(verification_results_path):
        try:
            with open(verification_results_path) as f:
                ver_data = json.load(f)
            if isinstance(ver_data, dict) and isinstance(ver_data.get("models"), dict):
                ver_models = ver_data["models"]
        except (OSError, ValueError):
            ver_models = {}

    sweeps_by_model = {}
    if sweeps_dir and os.path.isdir(sweeps_dir):
        for fname in sorted(os.listdir(sweeps_dir)):
            if not fname.endswith(".sweep.json"):
                continue
            fpath = os.path.join(sweeps_dir, fname)
            try:
                with open(fpath) as f:
                    sdata = json.load(f)
            except (OSError, ValueError):
                continue
            if not isinstance(sdata, dict):
                continue
            mid = sdata.get("model_id")
            if not isinstance(mid, str) or not mid:
                continue
            prev = sweeps_by_model.get(mid)
            if prev is None or str(sdata.get("swept_at", "")) >= str(prev.get("swept_at", "")):
                sweeps_by_model[mid] = sdata

    updated_models = []
    promoted_to_tested = []

    for m in catalog.get("models", []):
        mid = m.get("id", "")
        hf_repo = m.get("huggingface", "")
        changed = False

        ver_rec = ver_models.get(hf_repo) or ver_models.get(mid)
        if isinstance(ver_rec, dict) and ver_rec.get("verdict") == "SAFE":
            cur_ver = m.get("verification")
            if not isinstance(cur_ver, dict) or cur_ver.get("verdict") != "SAFE":
                m["verification"] = {
                    "verdict": "SAFE",
                    "verified_at": ver_rec.get("verified_at", ""),
                    "mlx_vlm_version": ver_rec.get("mlx_vlm_version", ""),
                    "notes": ver_rec.get(
                        "reason", "Admission gate verified via verify_max_kv_size.sh."
                    ),
                }
                changed = True

        sweep = sweeps_by_model.get(mid)
        if isinstance(sweep, dict):
            scf = sweep.get("derived_speed_correction_factor")
            if isinstance(scf, (int, float)) and not isinstance(scf, bool) and scf > 0:
                rounded_scf = round(float(scf), 3)
                if m.get("speed_correction_factor") != rounded_scf:
                    m["speed_correction_factor"] = rounded_scf
                    changed = True

            tc_rev = sweep.get("tool_calling_reverified")
            if isinstance(tc_rev, dict) and tc_rev.get("verdict") in ("SUPPORTED", "NOT_SUPPORTED"):
                cur_tc = (m.get("tool_calling") or {}).get("verdict")
                if cur_tc in (None, "UNTESTED"):
                    m["tool_calling"] = {
                        "verdict": tc_rev["verdict"],
                        "verified_at": sweep.get("swept_at", ""),
                        "method": (
                            "Live /v1/chat/completions tool-call tier via sweep_model.py "
                            "(temperature=0, web_search tool schema)."
                        ),
                    }
                    changed = True

        ver_safe = (m.get("verification") or {}).get("verdict") == "SAFE"
        tc_done = (m.get("tool_calling") or {}).get("verdict") in ("SUPPORTED", "NOT_SUPPORTED")
        if m.get("status") == "untested" and ver_safe and tc_done:
            m["status"] = "tested"
            disp = m.get("display")
            if isinstance(disp, dict):
                tagline = disp.get("tagline", "")
                if isinstance(tagline, str) and tagline.endswith(" — untested"):
                    disp["tagline"] = tagline[: -len(" — untested")]
            promoted_to_tested.append(mid)
            changed = True

        if changed:
            updated_models.append(mid)

    if write and updated_models:
        tmp_path = f"{models_path}.tmp"
        with open(tmp_path, "w") as f:
            f.write(json.dumps(catalog, indent=4, ensure_ascii=False) + "\n")
        os.replace(tmp_path, models_path)

    return {
        "updated_models": updated_models,
        "promoted_to_tested": promoted_to_tested,
        "catalog": catalog,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("query", nargs="?", help="repo id, Hugging Face URL, or model name to search for")
    ap.add_argument("--write", action="store_true", help="write changes to models.json instead of printing")
    ap.add_argument("--pick", type=int, help="for name searches: take the Nth result without asking")
    ap.add_argument("--id", help="override the generated id")
    ap.add_argument("--models-file", default=MODELS_JSON)
    ap.add_argument(
        "--sync-verified",
        action="store_true",
        help="sync docs/mlx_verification_results.json and ~/.bushpilot/calibration/sweeps/*.sweep.json into models.json",
    )
    ap.add_argument("--verification-file", default=VERIFICATION_RESULTS_JSON)
    ap.add_argument("--sweeps-dir", default=SWEEPS_DIR)
    args = ap.parse_args()

    if args.sync_verified:
        res = sync_verified_results(
            models_path=args.models_file,
            verification_results_path=args.verification_file,
            sweeps_dir=args.sweeps_dir,
            write=args.write,
        )
        summary = {
            "updated_models": res["updated_models"],
            "promoted_to_tested": res["promoted_to_tested"],
            "written": bool(args.write and res["updated_models"]),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if not args.query:
        ap.error("the following arguments are required: query (or pass --sync-verified)")

    repo = resolve_repo(args.query, args.pick)
    log(f"Reading {repo} ...")
    entry = build_entry(repo)
    if args.id:
        entry["id"] = args.id
    if args.write:
        write_entry(entry, args.models_file)
        log(f"Added '{entry['id']}' to {args.models_file} as untested "
            f"({entry['type']}, {entry['size_gb']:g} GB, {'+'.join(entry['modalities'])}).")
        log("Next: run verify_max_kv_size.sh and a tool-calling check, fill in display text, "
            "then set status to \"tested\".")
    else:
        print(json.dumps(entry, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
