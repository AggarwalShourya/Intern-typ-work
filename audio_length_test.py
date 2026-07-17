"""
audio_length_test.py — Find the maximum audio duration the server handles
                        at each concurrency level.

For every (duration, concurrency) cell it fires `concurrency` simultaneous
requests, classifies each response as ok / oom / error / crash, and prints:

  • A matrix of outcomes  (duration × concurrency)
  • Mean latency for cells that succeeded
  • Per-concurrency max safe duration summary

Use the results to decide what limits (if any) to enforce in production.

Usage:
    python audio_length_test.py [--url localhost:9020] [--audio overall.wav]
                                [--multipliers 1 2 3 6 12 30 60]
                                [--concurrencies 1 2 4 8]
                                [--reps 2]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Audio length vs. concurrency sweep")
parser.add_argument("--url",           default="localhost:9020")
parser.add_argument("--audio",         default="overall.wav")
parser.add_argument("--multipliers",   nargs="+", type=int,
                    default=[1, 2, 3, 6, 12, 18, 30, 60],
                    help="Repeat base audio N times to get longer clips")
parser.add_argument("--concurrencies", nargs="+", type=int,
                    default=[1, 2, 4, 8],
                    help="Number of simultaneous requests per cell")
parser.add_argument("--reps",          type=int, default=2,
                    help="Rounds of <concurrency> requests per cell "
                         "(total per cell = reps × concurrency)")
parser.add_argument("--cell-timeout",  type=float, default=300.0,
                    help="Seconds to wait for one cell before marking it as timeout")
parser.add_argument("--strategies",    nargs="+", default=["ctc", "rnnt"],
                    choices=["ctc", "rnnt", "malsd_batch", "tdt"],
                    help="Decoding strategies to sweep (each gets its own matrix)")
args = parser.parse_args()

MODEL_NAME  = "nemo_hybrid"
SAMPLE_RATE = 16000

# ── Client ────────────────────────────────────────────────────────────────────
client = grpcclient.InferenceServerClient(url=args.url, verbose=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_audio_int16(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except ImportError:
            raise RuntimeError(f"sr={sr} != {SAMPLE_RATE} and librosa not installed")
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


def set_decoding_strategy(strategy: str) -> None:
    _N = 8000
    dummy = np.zeros((1, _N), dtype=np.int16)
    inp_a = grpcclient.InferInput("AUDIO", [1, _N], "INT16")
    inp_a.set_data_from_numpy(dummy)
    inp_s = grpcclient.InferInput("SET_DECODING_STRATEGY", [1, 1], "BYTES")
    inp_s.set_data_from_numpy(np.array([[strategy.encode()]], dtype=object))
    outputs = [grpcclient.InferRequestedOutput(n)
               for n in ("TRANSCRIPT", "SCORE", "WORD_TIMESTAMPS")]
    client.infer(MODEL_NAME, [inp_a, inp_s], outputs=outputs)
    print(f"[client] Strategy → {strategy}")


def _infer_raw(audio: np.ndarray) -> Dict[str, Any]:
    """
    Single inference call.  Returns dict with keys:
      ok (bool), latency_ms (float), outcome (str), error_msg (str)

    outcome is one of: "ok" | "oom" | "error" | "crash"
    """
    audio_np = audio.reshape(1, -1)
    inp = grpcclient.InferInput("AUDIO", list(audio_np.shape), "INT16")
    inp.set_data_from_numpy(audio_np)
    outputs = [grpcclient.InferRequestedOutput(n)
               for n in ("TRANSCRIPT", "SCORE", "WORD_TIMESTAMPS")]

    t0 = time.perf_counter()
    try:
        client.infer(MODEL_NAME, [inp], outputs=outputs)
        ms = (time.perf_counter() - t0) * 1000.0
        return {"ok": True, "latency_ms": ms, "outcome": "ok", "error_msg": ""}
    except Exception as exc:
        ms  = (time.perf_counter() - t0) * 1000.0
        msg = str(exc)
        lo  = msg.lower()
        if "out of memory" in lo or "cuda" in lo or "oom" in lo or "alloc" in lo:
            outcome = "oom"
        elif "unavailable" in lo or "connection" in lo or "refused" in lo:
            outcome = "crash"
        else:
            outcome = "error"
        return {"ok": False, "latency_ms": ms, "outcome": outcome,
                "error_msg": msg[:200]}


def server_alive() -> bool:
    try:
        return client.is_server_ready() and client.is_model_ready(MODEL_NAME)
    except Exception:
        return False


# ── Cell measurement ──────────────────────────────────────────────────────────

def measure_cell(
    audio: np.ndarray,
    concurrency: int,
    reps: int,
) -> Dict[str, Any]:
    """
    Fire `reps` rounds of `concurrency` simultaneous requests.
    Returns aggregated stats for the cell.
    """
    results: List[Dict] = []

    for _ in range(reps):
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_infer_raw, audio) for _ in range(concurrency)]
            for f in as_completed(futures, timeout=args.cell_timeout):
                try:
                    results.append(f.result())
                except Exception as exc:
                    results.append({"ok": False, "latency_ms": float("nan"),
                                    "outcome": "crash", "error_msg": str(exc)[:200]})

    n_total   = len(results)
    n_ok      = sum(1 for r in results if r["ok"])
    outcomes  = [r["outcome"] for r in results]
    lats_ok   = [r["latency_ms"] for r in results if r["ok"]]
    errors    = [r["error_msg"] for r in results if not r["ok"]]

    # Dominant failure type (when not all ok)
    from collections import Counter
    fail_counts = Counter(o for o in outcomes if o != "ok")
    dominant_fail = fail_counts.most_common(1)[0][0] if fail_counts else None

    # Cell-level outcome classification
    rate = n_ok / max(n_total, 1)
    if rate >= 0.9:
        cell_outcome = "ok"
    elif rate >= 0.1:
        cell_outcome = "partial"
    elif dominant_fail == "oom":
        cell_outcome = "oom"
    elif dominant_fail == "crash":
        cell_outcome = "crash"
    else:
        cell_outcome = "error"

    mean_lat = sum(lats_ok) / len(lats_ok) if lats_ok else float("nan")

    return {
        "n_total":      n_total,
        "n_ok":         n_ok,
        "rate":         rate,
        "cell_outcome": cell_outcome,
        "mean_lat_ms":  mean_lat,
        "errors":       errors[:3],   # keep first 3 unique errors for display
    }


# ── Display ───────────────────────────────────────────────────────────────────

# Cell outcome → display symbol (for the matrix)
_SYMBOL = {
    "ok":      "OK",
    "partial": "PART",
    "oom":     "OOM",
    "error":   "ERR",
    "crash":   "CRASH",
    "?":       "?",
}

_COLOUR = {
    "ok":      "",        # no ANSI in plain output
    "partial": "",
    "oom":     "",
    "crash":   "",
}


def _cell_str(cell: Dict, show_lat: bool = True) -> str:
    sym = _SYMBOL.get(cell["cell_outcome"], "?")
    if cell["cell_outcome"] == "ok" and show_lat and not math.isnan(cell["mean_lat_ms"]):
        return f"{sym} {cell['mean_lat_ms']:>5.0f}ms"
    return f"{sym}"


def print_matrix(
    strategy: str,
    base_dur_s: float,
    multipliers: List[int],
    concurrencies: List[int],
    matrix: Dict[Tuple[int, int], Dict],
) -> None:
    """Print the (duration × concurrency) result matrix for one strategy."""
    col_w = 16
    dur_w = 9

    header = f"  {'duration':>{dur_w}}"
    for c in concurrencies:
        header += f"  {'conc='+str(c):>{col_w}}"
    sep = "─" * len(header)

    print(f"\n{'═'*len(header)}")
    print(f"  Audio length vs concurrency — strategy={strategy}  "
          f"(reps={args.reps}  total_per_cell=reps×conc)")
    print(f"  Base audio: {base_dur_s:.1f}s   "
          f"Outcomes: OK=all passed  PART=some passed  OOM=GPU OOM  "
          f"CRASH=server down  ERR=other")
    print(sep)
    print(header)
    print(sep)

    for mult in multipliers:
        dur_s = base_dur_s * mult
        row   = f"  {dur_s:>{dur_w-1}.0f}s"
        for c in concurrencies:
            cell = matrix.get((mult, c), {"cell_outcome": "?", "mean_lat_ms": float("nan")})
            row += f"  {_cell_str(cell):>{col_w}}"
        print(row)

    print(sep)


def _max_safe(
    multipliers: List[int],
    concurrencies: List[int],
    matrix: Dict[Tuple[int, int], Dict],
) -> Dict[int, Tuple[int, float]]:
    """Return {conc: (max_safe_mult, latency_ms)} for each concurrency."""
    result = {}
    for c in concurrencies:
        best_mult, best_lat = 0, float("nan")
        for mult in sorted(multipliers):
            cell = matrix.get((mult, c), {})
            if cell.get("cell_outcome") == "ok":
                best_mult = mult
                best_lat  = cell.get("mean_lat_ms", float("nan"))
        result[c] = (best_mult, best_lat)
    return result


def print_summary(
    strategy: str,
    base_dur_s: float,
    multipliers: List[int],
    concurrencies: List[int],
    matrix: Dict[Tuple[int, int], Dict],
) -> None:
    """Print max safe duration per concurrency level for one strategy."""
    safe = _max_safe(multipliers, concurrencies, matrix)
    any_crash = any(
        matrix.get((m, c), {}).get("cell_outcome") == "crash"
        for m in multipliers for c in concurrencies
    )

    print(f"\n  [{strategy}] Max safe duration per concurrency:")
    print(f"  {'concurrency':>14}  {'max safe dur':>18}  {'latency':>10}")
    print("  " + "─" * 48)
    for c in concurrencies:
        mult, lat = safe[c]
        if mult == 0:
            dur_str = "none (even base fails)"
            lat_str = "—"
        else:
            dur_str = f"{base_dur_s * mult:.0f}s  (×{mult})"
            lat_str = f"{lat:.0f}ms" if not math.isnan(lat) else "—"
        print(f"  {c:>14}  {dur_str:>18}  {lat_str:>10}")
    print()
    if any_crash:
        print("  *** WARNING: server crashed — check server logs immediately ***")
    else:
        print("  Server survived all cells ✓")


def print_comparison(
    base_dur_s: float,
    multipliers: List[int],
    concurrencies: List[int],
    all_matrices: Dict[str, Dict[Tuple[int, int], Dict]],
) -> None:
    """Side-by-side max safe duration comparison across all strategies."""
    strategies = list(all_matrices.keys())
    if len(strategies) < 2:
        return

    col_w = 20
    header = f"\n  {'concurrency':>14}"
    for s in strategies:
        header += f"  {s:>{col_w}}"
    header += f"  {'faster/longer':>14}"
    sep = "─" * len(header)

    print(f"\n{'█'*len(sep)}")
    print(f"  COMPARISON — Max safe audio duration  "
          f"[{' vs '.join(strategies)}]")
    print(f"{'█'*len(sep)}")
    print(header)
    print(sep)

    for c in concurrencies:
        row  = f"  {c:>14}"
        durs = []
        for s in strategies:
            mult, lat = _max_safe(multipliers, concurrencies, all_matrices[s])[c]
            dur_s = base_dur_s * mult if mult else 0.0
            durs.append(dur_s)
            cell_str = (f"{dur_s:.0f}s  ({lat:.0f}ms)" if mult and not math.isnan(lat)
                        else ("none" if mult == 0 else f"{dur_s:.0f}s"))
            row += f"  {cell_str:>{col_w}}"

        # Which strategy handles longer audio?
        if len(durs) == 2 and durs[0] != durs[1]:
            winner = strategies[0] if durs[0] > durs[1] else strategies[1]
            row += f"  {winner:>14}"
        elif len(durs) == 2:
            row += f"  {'tie':>14}"
        print(row)

    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_sweep(
    strategy: str,
    base_audio: np.ndarray,
    base_dur_s: float,
) -> Optional[Dict[Tuple[int, int], Dict]]:
    """Run the full (duration × concurrency) sweep for one strategy.
    Returns the matrix, or None if the server crashed mid-sweep."""
    set_decoding_strategy(strategy)
    time.sleep(0.5)

    multipliers   = sorted(args.multipliers)
    concurrencies = sorted(args.concurrencies)
    total_cells   = len(multipliers) * len(concurrencies)

    print(f"\n  Sweep: {len(multipliers)} durations × {len(concurrencies)} "
          f"concurrencies = {total_cells} cells  (reps={args.reps})\n")

    matrix: Dict[Tuple[int, int], Dict] = {}
    cell_num = 0

    for mult in multipliers:
        audio_long = np.tile(base_audio, mult)
        dur_s      = len(audio_long) / SAMPLE_RATE

        for conc in concurrencies:
            cell_num += 1
            n_req = args.reps * conc
            print(f"  [{cell_num:>2}/{total_cells}] "
                  f"dur={dur_s:>6.0f}s  conc={conc}  "
                  f"({n_req} requests) ... ", end="", flush=True)

            cell = measure_cell(audio_long, conc, args.reps)
            matrix[(mult, conc)] = cell

            lat_str = (f"{cell['mean_lat_ms']:.0f}ms"
                       if not math.isnan(cell["mean_lat_ms"]) else "—")
            print(f"{cell['cell_outcome'].upper():<8}  "
                  f"ok={cell['n_ok']}/{cell['n_total']}  lat={lat_str}")

            if cell["errors"]:
                print(f"           └ {cell['errors'][0][:100]}")

            if cell["cell_outcome"] == "crash" and not server_alive():
                print("\n  *** SERVER CRASHED — aborting sweep ***")
                print_matrix(strategy, base_dur_s, multipliers, concurrencies, matrix)
                print_summary(strategy, base_dur_s, multipliers, concurrencies, matrix)
                return None

    print()
    print_matrix(strategy, base_dur_s, multipliers, concurrencies, matrix)
    print_summary(strategy, base_dur_s, multipliers, concurrencies, matrix)
    return matrix


def main() -> None:
    assert server_alive(), "Server not ready — is Triton running?"
    print(f"Server OK  ({args.url})\n")

    base_audio = load_audio_int16(args.audio)
    base_dur_s = len(base_audio) / SAMPLE_RATE
    print(f"Base audio: {args.audio}  dur={base_dur_s:.2f}s  "
          f"samples={len(base_audio)}")
    print(f"Strategies : {args.strategies}\n")

    all_matrices: Dict[str, Dict[Tuple[int, int], Dict]] = {}

    for strategy in args.strategies:
        print(f"\n{'#'*70}")
        print(f"#  Strategy: {strategy}")
        print(f"{'#'*70}")
        matrix = run_sweep(strategy, base_audio, base_dur_s)
        if matrix is None:
            print(f"  Sweep aborted for {strategy} — server crashed.")
            break
        all_matrices[strategy] = matrix

    if len(all_matrices) > 1:
        print_comparison(
            base_dur_s,
            sorted(args.multipliers),
            sorted(args.concurrencies),
            all_matrices,
        )


if __name__ == "__main__":
    main()