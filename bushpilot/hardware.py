"""Hardware detection: chip name/GPU cores, total RAM, and currently-available
memory (the kernel's own real-time percent-free figure -- see
get_current_wired_gb's docstring for why)."""
import subprocess
import sys


def log(msg):
    print(msg, file=sys.stderr)


def get_current_wired_gb():
    """Returns memory currently UNAVAILABLE to a new model process, so the caller can
    compute free_gb = total_ram_gb - this value. Despite the historical name (kept so
    every call site doesn't change), this is not just "wired" memory.

    REVISED 2026-09-30, after two wrong turns documented here so the next person
    doesn't repeat either:
      1. The original implementation counted ONLY `Pages wired down` as used. Its real
         flaw was ignoring ACTIVE app memory and the compressor -- a machine with 15GB
         of running applications looked identical to an idle one, so a 14GB model
         could be recommended on top of a genuinely busy machine.
      2. A first rewrite then treated macOS's INACTIVE pages (its deliberately-kept
         file cache, which the kernel evicts on demand) as unavailable too. That
         overcorrected badly: an idle 24GB Mac measured only ~12.5GB free and the
         picker greyed out models that demonstrably run (gemma-4-26b ran fine for 14
         minutes on this machine at 65K context) -- a user-visible regression.

    The measure used now is the kernel's own real-time percent-free figure
    (`kern.memorystatus_level`, 0-100) -- the SAME number the launcher's memory
    backstop watches and that `memory_pressure(1)` reports as "System-wide memory free
    percentage". It is the honest middle: evictable cache counts as available (so
    recommendations match what actually runs, per the successful 26B run above), while
    ACTIVE app / compressor / wired consumption genuinely reduces it (so a busy
    machine is correctly seen as busy, fixing the original flaw). Because the backstop
    tears down on this same figure, the fit check and the guard can never disagree
    about what "free" means.

    Fallback (sysctl unavailable, e.g. a future macOS renaming it): vm_stat's
    free+speculative+purgeable+inactive as available -- closer to the kernel view than
    wired-only, still counts cache as reclaimable."""
    total_ram_gb = _get_total_ram_gb()
    if total_ram_gb is None:
        raise RuntimeError("Could not determine total RAM via sysctl hw.memsize")

    try:
        out = subprocess.check_output(["sysctl", "-n", "kern.memorystatus_level"], text=True)
        pct = int(out.strip())
        if 0 <= pct <= 100:
            return total_ram_gb * (100 - pct) / 100
    except (subprocess.CalledProcessError, ValueError):
        pass

    out = subprocess.check_output(["vm_stat"], text=True)
    page_size = 4096
    header = out.splitlines()[0]
    if "page size of" in header:
        page_size = int(header.split("page size of")[1].strip().split()[0])

    pages = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        try:
            pages[key.strip()] = int(val.strip().rstrip("."))
        except ValueError:
            continue

    reclaimable_keys = ("Pages free", "Pages speculative", "Pages purgeable", "Pages inactive")
    if not any(k in pages for k in reclaimable_keys):
        raise RuntimeError("Could not determine free memory from kern.memorystatus_level or vm_stat")
    reclaimable_gb = sum(pages.get(k, 0) for k in reclaimable_keys) * page_size / 1024**3
    return max(0.0, total_ram_gb - reclaimable_gb)


def _get_total_ram_gb():
    """Total physical RAM in GB via sysctl (fast, exact) -- separate from
    get_chip_info()'s system_profiler parse so this accounting doesn't depend on it."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / 1024**3
    except (subprocess.CalledProcessError, ValueError):
        return None


def get_chip_info():
    """Confirmed live: system_profiler SPHardwareDataType SPDisplaysDataType completes
    in ~0.45s on this machine — fast enough to run on every launch, no caching needed."""
    out = subprocess.check_output(
        ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"], text=True
    )
    chip = None
    total_ram_gb = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Chip:"):
            chip = s.split(":", 1)[1].strip()
        elif s.startswith("Memory:"):
            try:
                total_ram_gb = float(s.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    # "Total Number of Cores" appears twice: once under Hardware (CPU), once under
    # Displays (GPU). Confirmed live via the exact output this produced on this machine.
    core_lines = [line.strip() for line in out.splitlines() if "Total Number of Cores" in line]
    gpu_cores = None
    if len(core_lines) >= 2:
        try:
            gpu_cores = int(core_lines[1].split(":")[1].strip().split()[0])
        except (ValueError, IndexError):
            pass
    return chip, gpu_cores, total_ram_gb
