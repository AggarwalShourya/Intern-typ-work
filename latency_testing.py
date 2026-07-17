"""
latency_testing.py — Comprehensive latency benchmarking for the nemo_hybrid Triton server.

Tests all inference config paths under four sections:
  1. Sequential  — one request at a time (clean P50/P95/P99 per scenario)
  2. Concurrent  — N workers firing simultaneously (throughput + latency under load)
  3. Poisson     — requests arrive at a random rate matching real-world traffic
  4. Stress      — large burst of requests; checks queueing vs crashing

Each test section runs for every strategy and a side-by-side CTC vs transducer
comparison table is printed at the end of each section.

Usage:
    python latency_testing.py [--url localhost:9020] [--audio overall.wav]
                              [--runs 15] [--warmup 3]
                              [--stress-n 200] [--stress-conc 100]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import threading
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as grpcclient_aio

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="nemo_hybrid latency benchmark")
parser.add_argument("--url",         default="localhost:9020", help="Triton gRPC URL")
parser.add_argument("--audio",       default="overall.wav",    help="Test audio file")
parser.add_argument("--runs",        type=int, default=15,     help="Sequential runs per scenario")
parser.add_argument("--warmup",      type=int, default=3,      help="Warm-up runs (discarded)")
parser.add_argument("--keywords",    nargs="+", default=["keyword", "boosting"],
                    help="Keywords for context-graph tests")
parser.add_argument("--conc-levels", nargs="+", type=int, default=[2, 4, 8],
                    help="Concurrency levels to test")
parser.add_argument("--conc-total",  type=int, default=24,
                    help="Total requests per concurrency level")
parser.add_argument("--poisson-rate",  type=float, default=4.0,
                    help="Poisson test: average requests/sec")
parser.add_argument("--poisson-dur",   type=float, default=12.0,
                    help="Poisson test: wall-clock duration in seconds")
parser.add_argument("--poisson-drain", type=float, default=60.0,
                    help="Poisson test: max seconds to wait for in-flight requests "
                         "after the submission window closes (0 = wait forever)")
parser.add_argument("--stress-n",    type=int, default=200,
                    help="Stress test: total requests to fire")
parser.add_argument("--stress-conc", type=int, default=100,
                    help="Stress test: number of simultaneous workers")
parser.add_argument("--long-multipliers", nargs="+", type=int, default=[6, 60],
                    help="Long-audio test: repeat base audio N times (6=~1 min, 60=~10 min)")
# ── Transducer selection ──────────────────────────────────────────────────
# A single NeMo model supports either RNNT or TDT, not both.
# Change this variable to match the model you have deployed.
# CTC is always tested (both model types expose a CTC head).
TRANSDUCER = "tdt"          # ← switch to "tdt" for TDT models
# ──────────────────────────────────────────────────────────────────────────
parser.add_argument("--strategies",  nargs="+", default=["ctc", TRANSDUCER],
                    help="Decoding strategies to benchmark")
parser.add_argument("--dur-levels",  nargs="+", type=float,
                    default=[2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
                    help="Multi-duration batch test: clip durations in seconds")
parser.add_argument("--dur-conc",    type=int, default=8,
                    help="Multi-duration batch test: simultaneous workers per round")
parser.add_argument("--poisson-mixed-dur", action="store_true", default=True,
                    help="Poisson test: randomize audio duration per request "
                         "(realistic production traffic). Disable with --no-poisson-mixed-dur")
parser.add_argument("--no-poisson-mixed-dur", dest="poisson_mixed_dur", action="store_false")
parser.add_argument("--poisson-dur-range", nargs=2, type=float, default=[2.0, 15.0],
                    metavar=("MIN_S", "MAX_S"),
                    help="Poisson mixed-duration range in seconds (default: 2.0 15.0)")
args = parser.parse_args()

SERVER_URL  = args.url
AUDIO_FILE  = args.audio
MODEL_NAME  = "nemo_hybrid"
GRAPH_MODEL = "context_graph_builder"
SAMPLE_RATE = 16000
KEYWORDS    = args.keywords
N_WARMUP    = args.warmup
N_RUNS      = args.runs

_W_LABEL = 26
_W_NUM   = 8

# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio_int16(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            raise RuntimeError(
                f"Audio sample rate {sr} != {target_sr} and librosa is not installed."
            )
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


# ── Triton client ─────────────────────────────────────────────────────────────

client     = grpcclient.InferenceServerClient(url=SERVER_URL, verbose=False)
aio_client = grpcclient_aio.InferenceServerClient(url=SERVER_URL)


# ── Server helpers ────────────────────────────────────────────────────────────

def build_context_graph(keywords: List[str]) -> bytes:
    kw_json = json.dumps(keywords).encode()
    inp = grpcclient.InferInput("KEYWORDS", [1], "BYTES")
    inp.set_data_from_numpy(np.array([kw_json], dtype=object))
    out = grpcclient.InferRequestedOutput("CONTEXT_GRAPH")
    result = client.infer(GRAPH_MODEL, [inp], outputs=[out])
    return bytes(result.as_numpy("CONTEXT_GRAPH")[0])


def set_decoding_strategy(strategy: str) -> None:
    """
    Switch the server's decoding strategy at runtime.
    Sends a 0.5 s silent audio clip alongside SET_DECODING_STRATEGY.
    Valid values: "ctc", "rnnt", "malsd_batch", "tdt"
    """
    _N = 8000  # 0.5 s — must exceed mel-filterbank conv padding (>512 samples)
    dummy = np.zeros((1, _N), dtype=np.int16)
    inp_audio = grpcclient.InferInput("AUDIO", [1, _N], "INT16")
    inp_audio.set_data_from_numpy(dummy)

    inp_strat = grpcclient.InferInput("SET_DECODING_STRATEGY", [1, 1], "BYTES")
    inp_strat.set_data_from_numpy(np.array([[strategy.encode()]], dtype=object))

    outputs = [
        grpcclient.InferRequestedOutput("TRANSCRIPT"),
        grpcclient.InferRequestedOutput("SCORE"),
        grpcclient.InferRequestedOutput("WORD_TIMESTAMPS"),
    ]
    client.infer(MODEL_NAME, [inp_audio, inp_strat], outputs=outputs)
    print(f"[client] Decoding strategy → {strategy}")


def infer(
    audio:             np.ndarray,
    lang_tags:         Optional[List[str]] = None,
    blank_penalty:     float               = 0.0,
    cb_params:         Optional[Dict]      = None,
    context_graph:     Optional[bytes]     = None,
    word_boost_params: Optional[Dict]      = None,
    timestamps:        bool                = False,
    use_masked_joint:  bool                = False,
    timeout_s:         Optional[float]     = None,
) -> Dict[str, Any]:
    inputs, outputs = _build_inputs_outputs(
        audio, lang_tags, blank_penalty, cb_params,
        context_graph, word_boost_params, timestamps, use_masked_joint,
    )
    result = client.infer(MODEL_NAME, inputs, outputs=outputs, client_timeout=timeout_s)
    return _parse_result(result)


def _build_inputs_outputs(
    audio:             np.ndarray,
    lang_tags:         Optional[List[str]],
    blank_penalty:     float,
    cb_params:         Optional[Dict],
    context_graph:     Optional[bytes],
    word_boost_params: Optional[Dict],
    timestamps:        bool,
    use_masked_joint:  bool,
):
    """Shared input/output construction used by both async variants."""
    inputs = []
    outputs = [
        grpcclient.InferRequestedOutput("TRANSCRIPT"),
        grpcclient.InferRequestedOutput("SCORE"),
        grpcclient.InferRequestedOutput("WORD_TIMESTAMPS"),
    ]

    audio_np = audio.reshape(1, -1)
    inp = grpcclient.InferInput("AUDIO", list(audio_np.shape), "INT16")
    inp.set_data_from_numpy(audio_np)
    inputs.append(inp)

    if lang_tags is not None:
        val = np.array([[json.dumps(lang_tags).encode()]], dtype=object)
        inp = grpcclient.InferInput("LANG_TAGS", [1, 1], "BYTES")
        inp.set_data_from_numpy(val)
        inputs.append(inp)

    if blank_penalty != 0.0:
        val = np.array([[blank_penalty]], dtype=np.float32)
        inp = grpcclient.InferInput("BLANK_PENALTY", [1, 1], "FP32")
        inp.set_data_from_numpy(val)
        inputs.append(inp)

    if cb_params is not None:
        val = np.array([[json.dumps(cb_params).encode()]], dtype=object)
        inp = grpcclient.InferInput("CB_PARAMS", [1, 1], "BYTES")
        inp.set_data_from_numpy(val)
        inputs.append(inp)

    if context_graph is not None:
        val = np.array([[context_graph]], dtype=object)
        inp = grpcclient.InferInput("CONTEXT_GRAPH", [1, 1], "BYTES")
        inp.set_data_from_numpy(val)
        inputs.append(inp)

    if word_boost_params is not None:
        val = np.array([[json.dumps(word_boost_params).encode()]], dtype=object)
        inp = grpcclient.InferInput("WORD_BOOST_PARAMS", [1, 1], "BYTES")
        inp.set_data_from_numpy(val)
        inputs.append(inp)

    if timestamps:
        val = np.array([[True]], dtype=bool)
        inp = grpcclient.InferInput("TIMESTAMPS", [1, 1], "BOOL")
        inp.set_data_from_numpy(val)
        inputs.append(inp)

    if use_masked_joint:
        val = np.array([[True]], dtype=bool)
        inp = grpcclient.InferInput("USE_MASKED_JOINT", [1, 1], "BOOL")
        inp.set_data_from_numpy(val)
        inputs.append(inp)

    return inputs, outputs


def _parse_result(result) -> Dict[str, Any]:
    """Shared result extraction used by both async variants."""
    transcript = result.as_numpy("TRANSCRIPT")[0, 0]
    if isinstance(transcript, bytes):
        transcript = transcript.decode("utf-8")
    score = float(result.as_numpy("SCORE")[0, 0])
    wt_raw = result.as_numpy("WORD_TIMESTAMPS")[0, 0]
    if isinstance(wt_raw, bytes):
        wt_raw = wt_raw.decode("utf-8")
    word_timestamps = json.loads(wt_raw) if wt_raw else []
    return {"transcript": transcript, "score": score, "word_timestamps": word_timestamps}


# ── Async variant 1: native asyncio (tritonclient.grpc.aio) ───────────────────
# Use this when your calling code is already in an async context (FastAPI, etc.)
# Each await yields control back to the event loop so other coroutines can run
# concurrently while the server processes the request.
#
# Usage:
#   result = await infer_async(audio_int16, lang_tags=["hi"])
#
# Concurrent batch example:
#   results = await asyncio.gather(
#       infer_async(clip1, lang_tags=["hi"]),
#       infer_async(clip2, lang_tags=["te"]),
#       infer_async(clip3),
#   )

async def infer_async(
    audio:             np.ndarray,
    lang_tags:         Optional[List[str]] = None,
    blank_penalty:     float               = 0.0,
    cb_params:         Optional[Dict]      = None,
    context_graph:     Optional[bytes]     = None,
    word_boost_params: Optional[Dict]      = None,
    timestamps:        bool                = False,
    use_masked_joint:  bool                = False,
    timeout_s:         Optional[float]     = None,
) -> Dict[str, Any]:
    inputs, outputs = _build_inputs_outputs(
        audio, lang_tags, blank_penalty, cb_params,
        context_graph, word_boost_params, timestamps, use_masked_joint,
    )
    result = await aio_client.infer(
        MODEL_NAME, inputs, outputs=outputs, client_timeout=timeout_s
    )
    return _parse_result(result)


# ── Async variant 2: concurrent.futures.Future (callback-based) ───────────────
# Use this when your calling code is synchronous/threaded and you want to fire
# a request without blocking, then collect the result later.
#
# Usage — fire and forget, collect later:
#   fut = infer_future(audio_int16, lang_tags=["hi"])
#   # ... do other work ...
#   result = fut.result()          # blocks here until done
#
# Usage — send many requests in parallel:
#   futures = [infer_future(clip, lang_tags=["hi"]) for clip in clips]
#   results = [f.result() for f in concurrent.futures.as_completed(futures)]
#
# Usage — bridge into asyncio if needed:
#   result = await asyncio.wrap_future(infer_future(audio_int16))

def infer_future(
    audio:             np.ndarray,
    lang_tags:         Optional[List[str]] = None,
    blank_penalty:     float               = 0.0,
    cb_params:         Optional[Dict]      = None,
    context_graph:     Optional[bytes]     = None,
    word_boost_params: Optional[Dict]      = None,
    timestamps:        bool                = False,
    use_masked_joint:  bool                = False,
    timeout_s:         Optional[float]     = None,
) -> "concurrent.futures.Future[Dict[str, Any]]":
    inputs, outputs = _build_inputs_outputs(
        audio, lang_tags, blank_penalty, cb_params,
        context_graph, word_boost_params, timestamps, use_masked_joint,
    )
    future: concurrent.futures.Future = concurrent.futures.Future()

    def _callback(result, error):
        if error is not None:
            future.set_exception(error)
        else:
            try:
                future.set_result(_parse_result(result))
            except Exception as exc:
                future.set_exception(exc)

    client.async_infer(
        MODEL_NAME, inputs, callback=_callback,
        outputs=outputs, client_timeout=timeout_s,
    )
    return future


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {}
    s = sorted(latencies_ms)
    n = len(s)
    return {
        "n":    n,
        "mean": statistics.mean(s),
        "std":  statistics.stdev(s) if n > 1 else 0.0,
        "min":  s[0],
        "p50":  s[int(n * 0.50)],
        "p75":  s[int(n * 0.75)],
        "p95":  s[min(int(n * 0.95), n - 1)],
        "p99":  s[min(int(n * 0.99), n - 1)],
        "max":  s[-1],
    }


def rtf(latency_ms: float, audio_dur_s: float) -> float:
    return latency_ms / 1000.0 / max(audio_dur_s, 1e-9)


# ── Measurement drivers ───────────────────────────────────────────────────────

def measure_sequential(fn: Callable, n_warmup: int, n_runs: int) -> List[float]:
    for _ in range(n_warmup):
        fn()
    lats: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        lats.append((time.perf_counter() - t0) * 1000.0)
    return lats


def measure_concurrent(
    fn: Callable, concurrency: int, n_total: int
) -> Tuple[List[float], float]:
    def _timed():
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000.0

    lats: List[float] = []
    t_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for f in as_completed([pool.submit(_timed) for _ in range(n_total)]):
            lats.append(f.result())
    return lats, time.perf_counter() - t_wall


def measure_poisson(
    fn: Callable,
    rate_rps: float,
    duration_s: float,
    clip_pool: Optional[List[Tuple[float, np.ndarray]]] = None,
    infer_kwargs: Optional[Dict] = None,
    timeout_s: Optional[float] = None,
) -> List[float]:
    """
    Dispatch requests at Poisson inter-arrival times; collect per-request latencies.

    The approach is correct for measuring real-world load behaviour:
      • Poisson inter-arrivals model stationary random traffic
      • Concurrent requests exercise Triton's dynamic batcher
      • Reported latency includes server queue time (intentional — shows head-of-line
        blocking and queuing under load, not just per-request GPU time)

    If clip_pool is provided, each request randomly selects an audio clip from
    the pool (simulating variable-length production traffic).  In this mode,
    infer_kwargs is passed as **kwargs to infer(), and fn is ignored.

    IMPORTANT: fn should have a client_timeout baked in (e.g. via infer(...,
    timeout_s=N)) so that in-flight threads self-terminate when the server is
    slow.  Without a timeout, pool.shutdown(wait=True) blocks indefinitely and
    orphaned threads continue to hold Triton queue slots, stalling all
    subsequent test scenarios.
    """
    lats: List[float] = []
    lock = threading.Lock()
    _kw = infer_kwargs or {}
    _n_clips = len(clip_pool) if clip_pool else 0
    _rng = np.random.default_rng()

    def _timed():
        t0 = time.perf_counter()
        try:
            if clip_pool:
                _, audio = clip_pool[_rng.integers(0, _n_clips)]
                infer(audio, timeout_s=timeout_s, **_kw)
            else:
                fn()
        except Exception:
            pass    # timed-out / errored requests are not latency samples
        with lock:
            lats.append((time.perf_counter() - t0) * 1000.0)

    deadline = time.perf_counter() + duration_s
    pool     = ThreadPoolExecutor(max_workers=64)
    while time.perf_counter() < deadline:
        time.sleep(np.random.exponential(1.0 / rate_rps))
        if time.perf_counter() >= deadline:
            break
        pool.submit(_timed)
    pool.shutdown(wait=True)   # threads exit when client_timeout fires in fn()
    return lats


def measure_stress(
    fn: Callable, n_total: int, concurrency: int
) -> Dict[str, Any]:
    """
    Fire *n_total* requests with *concurrency* workers simultaneously.

    Tracks successes, failures, and error messages to determine whether the
    server queues excess load gracefully or crashes/rejects requests.

    Returns a dict with keys:
      n_sent, n_ok, n_failed, errors (Counter of message→count),
      latencies_ms (successful only), wall_s, server_alive_after
    """
    from collections import Counter

    lats:   List[float] = []
    errors: Counter     = Counter()
    lock    = threading.Lock()

    def _worker():
        t0 = time.perf_counter()
        try:
            fn()
            ms = (time.perf_counter() - t0) * 1000.0
            with lock:
                lats.append(ms)
        except Exception as exc:
            # Normalise error message: strip long stack detail, keep first line
            msg = str(exc).splitlines()[0][:120]
            with lock:
                errors[msg] += 1

    t_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker) for _ in range(n_total)]
        for f in futures:
            f.result()   # drain (exceptions already captured inside _worker)
    wall_s = time.perf_counter() - t_wall

    # Check server health after the burst
    try:
        alive = client.is_server_ready()
    except Exception:
        alive = False

    return {
        "n_sent":             n_total,
        "n_ok":               len(lats),
        "n_failed":           sum(errors.values()),
        "errors":             dict(errors),
        "latencies_ms":       lats,
        "wall_s":             wall_s,
        "server_alive_after": alive,
    }


# ── Scenario definitions ──────────────────────────────────────────────────────

def build_scenarios(cg: bytes, include_masked_joint: bool = False) -> List[Tuple[str, Dict]]:
    scenarios = [
        ("baseline",            {}),
        ("lang_hi",             {"lang_tags": ["hi"], "blank_penalty": 3.0}),
        ("lang_hi_en",          {"lang_tags": ["hi", "en"], "blank_penalty": 3.0,"use_masked_joint": True}),
        ("lang_te",             {"lang_tags": ["te"], "blank_penalty": 4.8,"use_masked_joint": True}),
        ("lang_te_en",          {"lang_tags": ["te", "en"],"use_masked_joint": True}),
        ("blank_penalty_2",     {"blank_penalty": 2.0}),
        ("word_boost_default",  {"word_boost_params": {"enabled": True}}),
        ("word_boost_strong",   {"word_boost_params": {"enabled": True, "lang_boost": 2.5,
                                                        "blank_penalty": 0.4}}),
        ("word_boost_full",     {"word_boost_params": {"enabled": True, "lang_boost": 2.0,
                                                        "lang_penalty": 1.5, "blank_penalty": 0.4}}),
        ("lang+word_boost",     {"lang_tags": ["hi", "en"],"use_masked_joint": True,
                                  "word_boost_params": {"enabled": True, "lang_boost": 2.0,
                                                         "blank_penalty": 0.4}}),
        ("kw_boost_default",    {"cb_params": {}, "context_graph": cg}),
        ("kw_boost_strong",     {"cb_params": {"cb_weight": 3.0, "keyword_threshold": 4.0},
                                  "context_graph": cg}),
        ("kw_boost+lang_te_en", {"lang_tags": ["te", "en"], "cb_params": {"cb_weight": 2.5},
                                  "context_graph": cg,"use_masked_joint": True}),
        ("timestamps",          {"timestamps": True}),
        ("timestamps+lang",     {"lang_tags": ["hi", "en"], "timestamps": True,"use_masked_joint": True}),
        ("timestamps+wb",       {"word_boost_params": {"enabled": True, "lang_boost": 2.0,
                                                        "blank_penalty": 0.4},
                                  "timestamps": True}),
        ("full_pipeline",       {"lang_tags": ["te", "en"],
                                  "word_boost_params": {"enabled": True, "lang_boost": 2.0,
                                                         "lang_penalty": 1.5, "blank_penalty": 0.4},
                                  "cb_params": {"cb_weight": 2.5}, "context_graph": cg,
                                  "timestamps": True,
                                  "use_masked_joint": True}),
    ]
    if include_masked_joint:
        # During-decode joint masking — only meaningful on RNNT/TDT.
        # CTC already applies a hard -inf mask unconditionally; these would
        # produce identical results to the non-mj variants on CTC.
        scenarios += [
            ("mj_lang_hi",    {"lang_tags": ["hi"],        "blank_penalty": 3.0, "use_masked_joint": True}),
            ("mj_lang_hi_en", {"lang_tags": ["hi", "en"],  "blank_penalty": 3.0, "use_masked_joint": True}),
            ("mj_lang_te",    {"lang_tags": ["te"],         "blank_penalty": 3.0, "use_masked_joint": True}),
            ("mj_lang_te_en", {"lang_tags": ["te", "en"],                         "use_masked_joint": True}),
        ]
    return scenarios


POISSON_SCENARIOS_KEYS = [
    "baseline", "lang_hi_en", "word_boost_default", "kw_boost_default", "full_pipeline"
]


# ── Per-strategy detail tables ────────────────────────────────────────────────

def _sep(width: int) -> str:
    return "─" * width


def _fmt_transcript(txt: str, n: int = 35) -> str:
    """Truncate long transcripts: show first n + … + last n chars."""
    if not txt:
        return "(empty)"
    txt = txt.strip()
    if len(txt) <= n * 2 + 5:
        return txt
    return txt[:n] + " … " + txt[-n:]


def print_sequential_table(
    strategy: str,
    results: Dict[str, List[float]],
    dur_s: float,
    transcripts: Optional[Dict[str, str]] = None,
) -> None:
    cols = ["mean", "std", "p50", "p95", "p99", "min", "max", "RTF"]
    w    = _W_LABEL + 2 + (_W_NUM + 2) * len(cols)
    print(f"\n{'═'*w}")
    print(f"  Sequential — strategy={strategy}  (ms except RTF)")
    print(_sep(w))
    print(f"  {'Scenario':<{_W_LABEL}}" +
          "  ".join(f"{c:>{_W_NUM}}" for c in cols))
    print(_sep(w))
    for label, lats in results.items():
        st = compute_stats(lats)
        r  = rtf(st["mean"], dur_s)
        row = (f"  {label:<{_W_LABEL}}"
               f"{st['mean']:>{_W_NUM}.1f}  {st['std']:>{_W_NUM}.1f}  "
               f"{st['p50']:>{_W_NUM}.1f}  {st['p95']:>{_W_NUM}.1f}  "
               f"{st['p99']:>{_W_NUM}.1f}  {st['min']:>{_W_NUM}.1f}  "
               f"{st['max']:>{_W_NUM}.1f}  {r:>{_W_NUM}.3f}")
        print(row)
        if transcripts and label in transcripts:
            print(f"  {'':>{_W_LABEL}}  └ {_fmt_transcript(transcripts[label])!r}")
    print(_sep(w))


def print_concurrent_table(
    strategy: str,
    results: Dict[int, Tuple[List[float], float]],
    n_total: int,
    transcript: Optional[str] = None,
) -> None:
    w = 72
    print(f"\n{'═'*w}")
    print(f"  Concurrent — strategy={strategy}  scenario=baseline  n_total={n_total}")
    print(_sep(w))
    print(f"  {'conc':>6}  {'n':>5}  {'mean ms':>9}  {'p95 ms':>9}  "
          f"{'p99 ms':>9}  {'req/s':>10}")
    print(_sep(w))
    for conc, (lats, wall) in sorted(results.items()):
        st = compute_stats(lats)
        print(f"  {conc:>6}  {st['n']:>5}  {st['mean']:>9.1f}  "
              f"{st['p95']:>9.1f}  {st['p99']:>9.1f}  {len(lats)/wall:>10.2f}")
    print(_sep(w))
    if transcript is not None:
        print(f"  sample transcript: {_fmt_transcript(transcript)!r}")


def print_poisson_table(
    strategy: str,
    results: Dict[str, List[float]],
    dur_s: float,
    transcripts: Optional[Dict[str, str]] = None,
) -> None:
    w = 72
    print(f"\n{'═'*w}")
    print(f"  Poisson — strategy={strategy}  "
          f"rate={args.poisson_rate:.1f} req/s  dur={args.poisson_dur:.0f}s")
    print(_sep(w))
    print(f"  {'Scenario':<{_W_LABEL}}  {'n':>5}  {'mean ms':>9}  "
          f"{'p95 ms':>9}  {'p99 ms':>9}  {'RTF':>6}")
    print(_sep(w))
    for label, lats in results.items():
        if not lats:
            print(f"  {label:<{_W_LABEL}}  (no requests completed)")
            continue
        st = compute_stats(lats)
        print(f"  {label:<{_W_LABEL}}  {st['n']:>5}  {st['mean']:>9.1f}  "
              f"{st['p95']:>9.1f}  {st['p99']:>9.1f}  "
              f"{rtf(st['mean'], dur_s):>6.3f}")
        if transcripts and label in transcripts:
            print(f"  {'':>{_W_LABEL}}         └ {_fmt_transcript(transcripts[label])!r}")
    print(_sep(w))


# ── Cross-strategy comparison tables ─────────────────────────────────────────

def _cmp_header(strategies: List[str], extra_cols: List[str] = ()) -> str:
    parts = [f"  {'Scenario':<{_W_LABEL}}"]
    for s in strategies:
        parts.append(f"  {(s+' ms'):>14}")
    for c in extra_cols:
        parts.append(f"  {c:>12}")
    return "".join(parts)


def print_comparison_sequential(
    all_seq: Dict[str, Dict[str, List[float]]],
    dur_s: float,
) -> None:
    strategies = list(all_seq.keys())
    if len(strategies) < 2:
        return
    labels = list(all_seq[strategies[0]].keys())
    hdr = _cmp_header(strategies, ["delta ms", "faster"])
    sep = _sep(len(hdr))
    print(f"\n\n{'█'*len(sep)}")
    print(f"  COMPARISON — Sequential mean latency (ms)  [{' vs '.join(strategies)}]")
    print(f"{'█'*len(sep)}")
    print(hdr)
    print(sep)
    for label in labels:
        means = []
        row = f"  {label:<{_W_LABEL}}"
        for s in strategies:
            lats = all_seq[s].get(label, [])
            m    = statistics.mean(lats) if lats else float("nan")
            means.append(m)
            row += f"  {m:>14.1f}"
        if len(means) == 2 and not any(math.isnan(v) for v in means):
            delta  = means[1] - means[0]
            sign   = "+" if delta >= 0 else ""
            faster = strategies[0] if delta > 0 else (strategies[1] if delta < 0 else "tie")
            row   += f"  {sign}{delta:>11.1f}  {faster:>12}"
        print(row)
    print(sep)


def print_comparison_concurrent(
    all_conc: Dict[str, Dict[int, Tuple[List[float], float]]],
) -> None:
    strategies = list(all_conc.keys())
    if len(strategies) < 2:
        return
    all_conc_levels = sorted({c for s in all_conc.values() for c in s})
    w = 90
    print(f"\n\n{'█'*w}")
    print(f"  COMPARISON — Concurrent mean latency & throughput  [{' vs '.join(strategies)}]")
    print(f"{'█'*w}")

    # Build header dynamically: conc | (mean_ms, req/s) per strategy | delta mean
    hdr = f"  {'conc':>6}"
    for s in strategies:
        hdr += f"  {s+' mean':>12}  {s+' req/s':>12}"
    hdr += f"  {'Δ mean ms':>12}  {'faster':>10}"
    print(hdr)
    print(_sep(len(hdr)))
    for conc in all_conc_levels:
        row   = f"  {conc:>6}"
        means = []
        for s in strategies:
            if conc in all_conc[s]:
                lats, wall = all_conc[s][conc]
                st = compute_stats(lats)
                means.append(st["mean"])
                row += f"  {st['mean']:>12.1f}  {len(lats)/wall:>12.2f}"
            else:
                means.append(float("nan"))
                row += f"  {'n/a':>12}  {'n/a':>12}"
        if len(means) == 2 and not any(math.isnan(v) for v in means):
            delta  = means[1] - means[0]
            sign   = "+" if delta >= 0 else ""
            faster = strategies[0] if delta > 0 else (strategies[1] if delta < 0 else "tie")
            row   += f"  {sign}{delta:>11.1f}  {faster:>10}"
        print(row)
    print(_sep(len(hdr)))


def print_comparison_poisson(
    all_poi: Dict[str, Dict[str, List[float]]],
    dur_s: float,
) -> None:
    strategies = list(all_poi.keys())
    if len(strategies) < 2:
        return
    labels = list(all_poi[strategies[0]].keys())
    hdr = _cmp_header(strategies, ["delta ms", "faster"])
    sep = _sep(len(hdr))
    print(f"\n\n{'█'*len(sep)}")
    print(f"  COMPARISON — Poisson mean latency (ms)  [{' vs '.join(strategies)}]")
    print(f"{'█'*len(sep)}")
    print(hdr)
    print(sep)
    for label in labels:
        means = []
        row = f"  {label:<{_W_LABEL}}"
        for s in strategies:
            lats = all_poi[s].get(label, [])
            m    = statistics.mean(lats) if lats else float("nan")
            means.append(m)
            disp = f"{m:.1f}" if not math.isnan(m) else "n/a"
            row += f"  {disp:>14}"
        if len(means) == 2 and not any(math.isnan(v) for v in means):
            delta  = means[1] - means[0]
            sign   = "+" if delta >= 0 else ""
            faster = strategies[0] if delta > 0 else (strategies[1] if delta < 0 else "tie")
            row   += f"  {sign}{delta:>11.1f}  {faster:>12}"
        print(row)
    print(sep)


# ── Stress-test report ────────────────────────────────────────────────────────

def print_stress_report(strategy: str, r: Dict[str, Any], transcript: Optional[str] = None) -> None:
    w   = 72
    ok  = r["n_ok"]
    bad = r["n_failed"]
    tot = r["n_sent"]
    print(f"\n{'═'*w}")
    print(f"  Stress test — strategy={strategy}  "
          f"n={tot}  concurrency={args.stress_conc}")
    print(_sep(w))
    print(f"  Sent        : {tot}")
    print(f"  Succeeded   : {ok}  ({100*ok/tot:.1f}%)")
    print(f"  Failed      : {bad}  ({100*bad/tot:.1f}%)")
    print(f"  Wall time   : {r['wall_s']:.2f}s  "
          f"({ok/max(r['wall_s'],1e-9):.1f} successful req/s)")
    print(f"  Server alive after burst: {'YES ✓' if r['server_alive_after'] else 'NO — CRASHED ✗'}")
    if r["latencies_ms"]:
        st = compute_stats(r["latencies_ms"])
        print(f"  Latency (successful): mean={st['mean']:.1f}ms  "
              f"p95={st['p95']:.1f}ms  p99={st['p99']:.1f}ms  max={st['max']:.1f}ms")
    if r["errors"]:
        print(f"  Error breakdown:")
        for msg, cnt in sorted(r["errors"].items(), key=lambda x: -x[1]):
            category = "queue_full/timeout" if "queue" in msg.lower() or "timeout" in msg.lower() \
                       else "connection_error" if "unavailable" in msg.lower() \
                       else "other"
            print(f"    [{category}] ×{cnt}  {msg[:80]}")
    else:
        print("  Errors: none — server queued all requests successfully.")
    if transcript is not None:
        print(f"  Sample transcript: {_fmt_transcript(transcript)!r}")
    print(_sep(w))


# ── Long-audio / OOM resilience test ─────────────────────────────────────────

def measure_long_audio(audio: np.ndarray, multiplier: int) -> Dict[str, Any]:
    """
    Concatenate *audio* *multiplier* times and send it to the server.

    Records the outcome in one of four categories:
      ok            — inference succeeded; returns latency_ms and transcript length
      rejected      — server returned TritonError containing "too long" / "MAX_AUDIO"
                      (the length guard in model.py fired before the GPU was touched)
      oom_caught    — server returned TritonError containing "out of memory"
                      (GPU OOM was caught inside _forward and returned as an error,
                       NOT a crash — server is still alive)
      crash         — server is unreachable / the call raised a connection error
                      (this should never happen after the model.py fixes)

    Also verifies the server is still alive after the call.
    """
    long_audio = np.tile(audio, multiplier)
    dur_s      = len(long_audio) / SAMPLE_RATE

    outcome    = "unknown"
    latency_ms = float("nan")
    transcript = ""
    error_msg  = ""

    t0 = time.perf_counter()
    try:
        result     = infer(long_audio)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        outcome    = "ok"
        transcript = result["transcript"]
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        error_msg  = str(exc)
        lo = error_msg.lower()
        if "too long" in lo or "max_audio" in lo or "split" in lo:
            outcome = "rejected"          # length guard fired — server healthy
        elif "out of memory" in lo or "cuda" in lo or "oom" in lo:
            outcome = "oom_caught"        # OOM caught by except in _forward
        else:
            outcome = "crash"             # unexpected / connection error

    # Check server health regardless of outcome
    try:
        alive = client.is_server_ready() and client.is_model_ready(MODEL_NAME)
    except Exception:
        alive = False
        if outcome == "unknown":
            outcome = "crash"

    return {
        "multiplier":    multiplier,
        "dur_s":         dur_s,
        "outcome":       outcome,
        "latency_ms":    latency_ms,
        "server_alive":  alive,
        "transcript":    transcript,
        "transcript_len": len(transcript),
        "error_msg":     error_msg[:200],
    }


_OUTCOME_LABEL = {
    "ok":         "OK       — inference succeeded",
    "rejected":   "REJECTED — length guard (safe, no GPU touched)",
    "oom_caught": "OOM      — GPU ran out of memory (caught, server alive)",
    "crash":      "CRASH    — server unreachable (CRITICAL ✗)",
}


def print_long_audio_report(strategy: str, results: List[Dict[str, Any]]) -> None:
    w = 80
    print(f"\n{'═'*w}")
    print(f"  Long-audio / OOM resilience — strategy={strategy}")
    print(_sep(w))
    print(f"  {'dur':>8}  {'mult':>5}  {'latency ms':>12}  "
          f"{'server alive':>13}  outcome")
    print(_sep(w))
    for r in results:
        dur_str = f"{r['dur_s']:.0f}s"
        lat_str = f"{r['latency_ms']:.0f}" if not math.isnan(r['latency_ms']) else "n/a"
        alive   = "YES ✓" if r["server_alive"] else "NO ✗"
        label   = _OUTCOME_LABEL.get(r["outcome"], r["outcome"])
        print(f"  {dur_str:>8}  {r['multiplier']:>5}x  {lat_str:>12}  "
              f"{alive:>13}  {label}")
        if r["outcome"] == "ok" and r.get("transcript"):
            print(f"  {'':<8}  {'':<5}   {'':<12}  {'':<13}  "
                  f"  └ {_fmt_transcript(r['transcript'])!r}")
        elif r["error_msg"]:
            print(f"  {'':<8}  {'':<5}   {'':<12}  {'':<13}  "
                  f"  └ {r['error_msg'][:90]}")
    print(_sep(w))
    # Summary verdict
    crashes = [r for r in results if r["outcome"] == "crash"]
    if crashes:
        print(f"  *** VERDICT: SERVER CRASHED on {len(crashes)} request(s) — "
              f"model.py needs OOM guard fixes ***")
    else:
        print(f"  VERDICT: Server survived all long-audio requests — "
              f"queueing/rejection working correctly ✓")
    print(_sep(w))


# ── Multi-duration batch test ─────────────────────────────────────────────────

def build_mixed_dur_pool(
    audio_int16: np.ndarray,
    dur_min: float,
    dur_max: float,
    n_clips: int = 20,
    sr: int = SAMPLE_RATE,
) -> List[Tuple[float, np.ndarray]]:
    """
    Pre-build a pool of clips at uniformly spaced durations between dur_min
    and dur_max.  The Poisson test picks randomly from this pool for each
    request, simulating realistic variable-length production traffic.

    Returns [(dur_s, audio_int16), ...] sorted by duration.
    """
    durations = np.linspace(dur_min, dur_max, n_clips)
    pool = []
    for d in durations:
        clip = make_clip(audio_int16, d, sr)
        pool.append((round(float(d), 2), clip))
    print(f"  [mixed-dur pool] {n_clips} clips from {dur_min:.1f}s to {dur_max:.1f}s")
    return pool


def make_clip(audio_int16: np.ndarray, dur_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Return exactly dur_s seconds of audio from audio_int16.
    Clips shorter than the base are sliced from the start.
    Clips longer tile the base audio then slice to the exact target length.
    """
    target = int(dur_s * sr)
    if target <= len(audio_int16):
        return audio_int16[:target]
    reps = math.ceil(target / len(audio_int16))
    return np.tile(audio_int16, reps)[:target]


def measure_multi_duration_solo(
    clips: List[Tuple[float, np.ndarray]],
    n_warmup: int = 1,
    n_runs: int = 3,
) -> Dict[float, Dict[str, Any]]:
    """
    Single-item reference run for each duration.
    Returns {dur_s → {latencies_ms, transcripts}}.
    """
    results: Dict[float, Dict[str, Any]] = {}
    for dur_s, audio in clips:
        fn = lambda a=audio: infer(a)
        for _ in range(n_warmup):
            fn()
        lats, txts = [], []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            r  = fn()
            lats.append((time.perf_counter() - t0) * 1000.0)
            txts.append(r["transcript"])
        results[dur_s] = {"latencies_ms": lats, "transcripts": txts}
    return results


def measure_multi_duration_concurrent(
    clips:    List[Tuple[float, np.ndarray]],
    n_rounds: int = 5,
) -> Dict[float, Dict[str, Any]]:
    """
    Each round fires one request per duration level simultaneously, so Triton
    sees a burst of requests with different audio lengths arriving together.
    The dynamic batcher groups them into a single execute() call (padded to T_max).

    Returns {dur_s → {latencies_ms, transcripts, errors}}.
    """
    results: Dict[float, Dict[str, Any]] = {
        dur: {"latencies_ms": [], "transcripts": [], "errors": 0}
        for dur, _ in clips
    }
    lock = threading.Lock()

    def _one(dur_s: float, audio: np.ndarray) -> None:
        t0 = time.perf_counter()
        try:
            r  = infer(audio)
            ms = (time.perf_counter() - t0) * 1000.0
            with lock:
                results[dur_s]["latencies_ms"].append(ms)
                results[dur_s]["transcripts"].append(r["transcript"])
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000.0
            with lock:
                results[dur_s]["errors"] += 1
                results[dur_s]["latencies_ms"].append(ms)
                results[dur_s]["transcripts"].append(f"ERROR: {exc}")

    for rnd in range(n_rounds):
        with ThreadPoolExecutor(max_workers=len(clips)) as pool:
            futures = [pool.submit(_one, dur, audio) for dur, audio in clips]
            for f in futures:
                f.result()
        print(f"    round {rnd+1}/{n_rounds} done", flush=True)

    return results


def _transcript_match(ref_txts: List[str], batch_txts: List[str]) -> str:
    """
    Compare the most common reference transcript against batch transcripts.
    Returns 'ok', 'mismatch', or 'error' (if all batch entries are errors).
    """
    if not ref_txts or not batch_txts:
        return "?"
    if all(t.startswith("ERROR:") for t in batch_txts):
        return "error"
    # Normalize: lowercase, collapse whitespace
    def _norm(t: str) -> str:
        return " ".join(t.lower().split())
    ref  = _norm(max(set(ref_txts), key=ref_txts.count))
    hits = sum(1 for t in batch_txts if _norm(t) == ref)
    return "ok" if hits == len(batch_txts) else f"~{hits}/{len(batch_txts)}"


def print_multi_duration_report(
    strategy:      str,
    clips:         List[Tuple[float, np.ndarray]],
    solo_results:  Dict[float, Dict[str, Any]],
    batch_results: Dict[float, Dict[str, Any]],
    threshold_s:   float,
) -> None:
    w = 90
    print(f"\n{'═'*w}")
    print(f"  Multi-duration batch test — strategy={strategy}  "
          f"threshold={threshold_s:.0f}s (above → chunked)")
    print(f"  Solo: 1 request at a time.  Concurrent: all durations fired simultaneously.")
    print(f"  'match': do all concurrent transcripts match the solo reference? "
          f"ok=all match  ~N/M=partial")
    print(f"─{(w-1)*'─'}")
    print(f"  {'dur':>5}  {'path':<8}  "
          f"{'solo mean':>10}  {'solo p95':>9}  {'solo RTF':>9}  "
          f"{'conc mean':>10}  {'conc p95':>9}  {'errors':>7}  {'match':>6}")
    print(f"─{(w-1)*'─'}")

    for dur_s, _ in clips:
        path  = "CHUNKED" if dur_s > threshold_s else "batch"
        sr    = solo_results.get(dur_s, {})
        br    = batch_results.get(dur_s, {})

        s_lats = sr.get("latencies_ms", [])
        b_lats = [x for x in br.get("latencies_ms", []) if not math.isnan(x)]

        s_mean = statistics.mean(s_lats) if s_lats else float("nan")
        s_p95  = sorted(s_lats)[min(int(len(s_lats)*0.95), len(s_lats)-1)] if s_lats else float("nan")
        s_rtf  = s_mean / 1000.0 / dur_s if s_mean == s_mean else float("nan")
        b_mean = statistics.mean(b_lats) if b_lats else float("nan")
        b_p95  = sorted(b_lats)[min(int(len(b_lats)*0.95), len(b_lats)-1)] if b_lats else float("nan")
        errs   = br.get("errors", 0)

        match  = _transcript_match(
            sr.get("transcripts", []), br.get("transcripts", [])
        )

        def _fmt(v: float, fmt: str = ".1f") -> str:
            return f"{v:{fmt}}" if v == v else "n/a"

        print(f"  {dur_s:>4.0f}s  {path:<8}  "
              f"{_fmt(s_mean):>10}  {_fmt(s_p95):>9}  {_fmt(s_rtf, '.3f'):>9}  "
              f"{_fmt(b_mean):>10}  {_fmt(b_p95):>9}  {errs:>7}  {match:>6}")
        solo_txts = sr.get("transcripts", [])
        if solo_txts:
            print(f"  {'':>4}   {'':>8}    └ {_fmt_transcript(solo_txts[0])!r}")

    print(f"─{(w-1)*'─'}")
    all_errs = sum(br.get("errors", 0) for br in batch_results.values())
    if all_errs:
        print(f"  *** {all_errs} error(s) in concurrent run — check server logs ***")
    else:
        print(f"  All durations completed without errors in concurrent run ✓")
    print(f"{'═'*w}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Connecting to {SERVER_URL} ...")
    assert client.is_server_ready(),           "Triton server not ready"
    assert client.is_model_ready(MODEL_NAME),  f"{MODEL_NAME} not ready"
    assert client.is_model_ready(GRAPH_MODEL), f"{GRAPH_MODEL} not ready"
    print("Server OK.\n")

    audio_int16 = load_audio_int16(AUDIO_FILE)
    dur_s       = len(audio_int16) / SAMPLE_RATE
    print(f"Audio: {AUDIO_FILE}  samples={len(audio_int16)}  duration={dur_s:.2f}s\n")

    print(f"Building context graph for keywords: {KEYWORDS} ...")
    cg = build_context_graph(KEYWORDS)
    print(f"  Context graph: {len(cg)} bytes\n")

    # masked_joint scenarios are RNNT/TDT-only; built once and reused across
    # strategies.  When the strategy loop hits CTC the mj_* entries are skipped
    # because build_scenarios is called with include_masked_joint=False for CTC.
    # Use a per-strategy build below.
    poisson_base      = build_scenarios(cg, include_masked_joint=False)
    poisson_scenarios = [(k, dict(s)) for k, s in poisson_base if k in POISSON_SCENARIOS_KEYS]

    # Accumulated results across strategies for comparison tables
    all_sequential: Dict[str, Dict[str, List[float]]]              = {}
    all_concurrent: Dict[str, Dict[int, Tuple[List[float], float]]] = {}
    all_poisson:    Dict[str, Dict[str, List[float]]]               = {}

    # ── Per-strategy loop ──────────────────────────────────────────────────────
    for strategy in args.strategies:
        print(f"\n{'#'*72}")
        print(f"#  Strategy: {strategy}")
        print(f"{'#'*72}")

        set_decoding_strategy(strategy)
        time.sleep(0.5)   # allow decoder reconfiguration to settle

        is_transducer = strategy != "ctc"
        scenarios = build_scenarios(cg, include_masked_joint=is_transducer)

        # ── 1. Sequential ──────────────────────────────────────────────────────
        print(f"\n[{strategy}] Sequential  "
              f"(warmup={N_WARMUP}  runs={N_RUNS}  scenarios={len(scenarios)})")
        seq: Dict[str, List[float]] = {}
        seq_transcripts: Dict[str, str] = {}
        for label, kwargs in scenarios:
            fn = lambda kw=kwargs: infer(audio_int16, **kw)
            print(f"  {label:<{_W_LABEL}} ", end="", flush=True)
            lats = measure_sequential(fn, N_WARMUP, N_RUNS)
            st   = compute_stats(lats)
            print(f"mean={st['mean']:.1f}ms  p95={st['p95']:.1f}ms  "
                  f"RTF={rtf(st['mean'], dur_s):.3f}")
            seq[label] = lats
            seq_transcripts[label] = infer(audio_int16, **kwargs)["transcript"]
        all_sequential[strategy] = seq
        print_sequential_table(strategy, seq, dur_s, seq_transcripts)

        # ── 2. Concurrent ──────────────────────────────────────────────────────
        print(f"\n[{strategy}] Concurrent  "
              f"(baseline  n_total={args.conc_total}  levels={args.conc_levels})")
        baseline_fn = lambda: infer(audio_int16)
        conc: Dict[int, Tuple[List[float], float]] = {}
        for c in args.conc_levels:
            print(f"  concurrency={c:<3} ", end="", flush=True)
            lats, wall = measure_concurrent(baseline_fn, c, args.conc_total)
            st = compute_stats(lats)
            print(f"mean={st['mean']:.1f}ms  p95={st['p95']:.1f}ms  "
                  f"throughput={len(lats)/wall:.2f} req/s")
            conc[c] = (lats, wall)
        all_concurrent[strategy] = conc
        conc_transcript = infer(audio_int16)["transcript"]
        print_concurrent_table(strategy, conc, args.conc_total, conc_transcript)

        # ── 3. Poisson ─────────────────────────────────────────────────────────
        _mixed = args.poisson_mixed_dur
        _mix_label = "mixed-dur" if _mixed else "fixed-dur"
        print(f"\n[{strategy}] Poisson ({_mix_label})  "
              f"(rate={args.poisson_rate:.1f} req/s  dur={args.poisson_dur:.0f}s)")

        # Build mixed-duration clip pool once per strategy
        _clip_pool = None
        _poi_dur_s = dur_s
        if _mixed:
            _dur_min, _dur_max = args.poisson_dur_range
            _clip_pool = build_mixed_dur_pool(audio_int16, _dur_min, _dur_max)
            # Report the mean clip duration for RTF calculation
            _poi_dur_s = sum(d for d, _ in _clip_pool) / len(_clip_pool)
            print(f"  mean clip duration for RTF: {_poi_dur_s:.2f}s")

        poi: Dict[str, List[float]] = {}
        poi_transcripts: Dict[str, str] = {}
        _poi_timeout = args.poisson_drain or None
        for label, kwargs in poisson_scenarios:
            print(f"  {label:<{_W_LABEL}} ", end="", flush=True)
            if _mixed and _clip_pool:
                lats = measure_poisson(
                    fn=None, rate_rps=args.poisson_rate,
                    duration_s=args.poisson_dur,
                    clip_pool=_clip_pool,
                    infer_kwargs=kwargs,
                    timeout_s=_poi_timeout,
                )
            else:
                # Bake the client timeout into fn so threads self-terminate after
                # poisson_drain seconds — prevents orphaned Triton queue slots from
                # stalling subsequent test scenarios.
                fn = lambda kw=kwargs, t=_poi_timeout: infer(audio_int16, timeout_s=t, **kw)
                lats = measure_poisson(fn, args.poisson_rate, args.poisson_dur)
            if lats:
                st = compute_stats(lats)
                print(f"n={st['n']}  mean={st['mean']:.1f}ms  p95={st['p95']:.1f}ms")
            else:
                print("(no requests completed)")
            poi[label] = lats
            poi_transcripts[label] = infer(audio_int16, **kwargs)["transcript"]
        all_poisson[strategy] = poi
        print_poisson_table(strategy, poi, _poi_dur_s, poi_transcripts)

        # ── 4. Stress ──────────────────────────────────────────────────────────
        print(f"\n[{strategy}] Stress  "
              f"(n={args.stress_n}  concurrency={args.stress_conc})")
        print(f"  Firing {args.stress_n} simultaneous requests ...", flush=True)
        stress_r = measure_stress(baseline_fn, args.stress_n, args.stress_conc)
        stress_transcript = infer(audio_int16)["transcript"]
        print_stress_report(strategy, stress_r, stress_transcript)

        # ── 5. Long-audio / OOM resilience ─────────────────────────────────────
        print(f"\n[{strategy}] Long-audio resilience  "
              f"(multipliers={args.long_multipliers}  base_dur={dur_s:.1f}s)")
        long_results: List[Dict[str, Any]] = []
        for mult in args.long_multipliers:
            target_dur = dur_s * mult
            print(f"  x{mult:<3} ({target_dur:.0f}s) ... ", end="", flush=True)
            r = measure_long_audio(audio_int16, mult)
            print(f"outcome={r['outcome']}  "
                  f"latency={r['latency_ms']:.0f}ms  "
                  f"server_alive={r['server_alive']}")
            long_results.append(r)
        print_long_audio_report(strategy, long_results)

        # ── 6. Multi-duration batch test ───────────────────────────────────────
        # Build clips for every requested duration level (clip or tile the base).
        # Durations that exceed the base audio length are covered by tiling.
        # Durations longer than LONG_AUDIO_THRESHOLD_S will go through the new
        # chunking path; shorter ones will be batched normally by Triton.
        _dur_levels = sorted(set(args.dur_levels))
        _clips: List[Tuple[float, np.ndarray]] = [
            (d, make_clip(audio_int16, d)) for d in _dur_levels
        ]
        print(
            f"\n[{strategy}] Multi-duration batch test  "
            f"durations={[f'{d:.0f}s' for d, _ in _clips]}  "
            f"conc={len(_clips)} (one per duration per round)  "
            f"n_rounds={args.dur_conc}"
        )

        # Solo reference (1 warmup, 3 runs each)
        print(f"  [solo reference] ", end="", flush=True)
        solo_res = measure_multi_duration_solo(_clips, n_warmup=1, n_runs=3)
        print("done")

        # Concurrent: all durations fired simultaneously, N rounds
        print(f"  [concurrent rounds] ", flush=True)
        conc_res = measure_multi_duration_concurrent(_clips, n_rounds=args.dur_conc)

        print_multi_duration_report(
            strategy, _clips, solo_res, conc_res,
            threshold_s=20.0,   # matches LONG_AUDIO_THRESHOLD_S default
        )

    # ── Cross-strategy comparison tables (printed once, at the end) ───────────
    if len(args.strategies) > 1:
        print(f"\n\n{'█'*72}")
        print(f"  OVERALL COMPARISON: {' vs '.join(args.strategies)}")
        print(f"{'█'*72}")

        print_comparison_sequential(all_sequential, dur_s)
        print_comparison_concurrent(all_concurrent)
        print_comparison_poisson(all_poisson, dur_s)


if __name__ == "__main__":
    main()
