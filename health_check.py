"""
health_check.py — Health & metrics endpoint for the nemo_hybrid Triton server.

No test inference.  model.py writes a JSON status file on every request and on
CUDA errors.  This script reads that file + polls Triton's own HTTP API.
Zero GPU overhead.

HTTP endpoints (port HEALTH_PORT, default 9005):
  GET /health/live   — liveness:  is the process alive and GPU not corrupted?
                        K8s restarts the pod on persistent 503.
  GET /health/ready  — readiness: is the model loaded and ready to serve?
                        K8s removes the pod from the LB pool on 503.
  GET /health        — full JSON status (backward-compat, same as readiness body)
  GET /metrics       — Prometheus text format.  Proxies Triton's own /metrics
                        (port TRITON_METRICS_PORT) and appends GPU/health gauges
                        derived from the status file.

CLI (Docker HEALTHCHECK / K8s exec probe):
  python health_check.py --check        → exit 0 if ready,    exit 1 if not
  python health_check.py --check-live   → exit 0 if live,     exit 1 if not

Environment variables:
  HEALTH_PORT           HTTP port for this server         (default 9005)
  TRITON_HTTP_URL       Triton HTTP address               (default localhost:9002)
  TRITON_METRICS_PORT   Triton Prometheus metrics port    (default 9003)
  MODEL_NAME            Triton model name                 (default nemo_hybrid)
  HEALTH_STATUS_FILE    Status file written by model.py   (default /tmp/nemo_hybrid_health.json)
  STALE_THRESHOLD_S     Seconds before GPU status is stale (default 300)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── Config ───────────────────────────────────────────────────────────────────

HEALTH_PORT          = int(os.environ.get("HEALTH_PORT", "9005"))
TRITON_HTTP_URL      = os.environ.get("TRITON_HTTP_URL", "localhost:9002")
TRITON_METRICS_PORT  = int(os.environ.get("TRITON_METRICS_PORT", "9003"))
MODEL_NAME           = os.environ.get("MODEL_NAME", "nemo_hybrid")
HEALTH_STATUS_FILE   = os.environ.get("HEALTH_STATUS_FILE", "/tmp/nemo_hybrid_health.json")
STALE_THRESHOLD_S    = int(os.environ.get("STALE_THRESHOLD_S", "300"))


# ── Triton checks ─────────────────────────────────────────────────────────────

def _triton_server_ready() -> bool:
    """True if the Triton HTTP server is responding."""
    try:
        with urllib.request.urlopen(
            f"http://{TRITON_HTTP_URL}/v2/health/ready", timeout=5
        ) as r:
            return r.status == 200
    except Exception:
        return False


def _triton_model_ready() -> bool:
    """True if our model is loaded and ready."""
    try:
        with urllib.request.urlopen(
            f"http://{TRITON_HTTP_URL}/v2/models/{MODEL_NAME}/ready", timeout=5
        ) as r:
            return r.status == 200
    except Exception:
        return False


def _fetch_triton_metrics() -> str | None:
    """Return the raw Prometheus text from Triton's /metrics endpoint, or None."""
    try:
        with urllib.request.urlopen(
            f"http://localhost:{TRITON_METRICS_PORT}/metrics", timeout=5
        ) as r:
            return r.read().decode("utf-8")
    except Exception:
        return None


# ── GPU / model.py status file ───────────────────────────────────────────────

def _read_gpu_status() -> dict:
    """
    Read the JSON status file written by model.py.
    On FileNotFoundError (server just started, no requests yet) we treat GPU
    as healthy and not stale — model.py has not had a chance to write yet.
    """
    try:
        with open(HEALTH_STATUS_FILE) as f:
            data = json.load(f)
        age = time.time() - data.get("timestamp_epoch", 0)
        stale = age > STALE_THRESHOLD_S
        return {
            "gpu_ok":          data.get("cuda_healthy", False),
            "stale":           stale,
            "age_s":           round(age, 1),
            "last_error":      data.get("last_error"),
            "last_updated":    data.get("timestamp"),
            "total_requests":  data.get("total_requests", 0),
            "failed_requests": data.get("failed_requests", 0),
        }
    except FileNotFoundError:
        # No requests have been processed yet — this is fine during startup.
        return {
            "gpu_ok":          True,
            "stale":           False,
            "age_s":           0,
            "last_error":      None,
            "last_updated":    None,
            "total_requests":  0,
            "failed_requests": 0,
        }
    except Exception as exc:
        return {
            "gpu_ok":          False,
            "stale":           True,
            "age_s":           -1,
            "last_error":      f"status file read error: {exc}",
            "last_updated":    None,
            "total_requests":  0,
            "failed_requests": 0,
        }


# ── Composite checks ──────────────────────────────────────────────────────────

def _liveness_check() -> dict:
    """
    Liveness: should K8s kill and restart this pod?
    Fails only on hard failures that won't self-heal:
      - Triton process is unreachable (crashed / hung)
      - GPU context is corrupted (CUDA error — requires pod restart to recover)
    Does NOT fail just because the model isn't loaded yet (that's readiness).
    """
    triton_up = _triton_server_ready()
    gpu       = _read_gpu_status()
    live      = triton_up and gpu["gpu_ok"]
    return {
        "live":          live,
        "triton_up":     triton_up,
        "gpu_ok":        gpu["gpu_ok"],
        "last_error":    gpu["last_error"],
        "last_updated":  gpu["last_updated"],
    }


def _readiness_check() -> dict:
    """
    Readiness: should K8s send traffic to this pod?
    Fails when:
      - Triton is not reachable
      - The model is not yet loaded
      - GPU context is corrupted
      - Status file has gone stale (model.py may be hung)
    """
    triton_up   = _triton_server_ready()
    model_ready = _triton_model_ready() if triton_up else False
    gpu         = _read_gpu_status()

    # A stale status file means model.py hasn't processed a request in
    # STALE_THRESHOLD_S.  Only treat as not-ready if we know requests have
    # been attempted (total_requests > 0); otherwise it's just a fresh pod.
    stale_and_active = gpu["stale"] and gpu["total_requests"] > 0

    ready = triton_up and model_ready and gpu["gpu_ok"] and not stale_and_active
    return {
        "ready":           ready,
        "triton_up":       triton_up,
        "model_ready":     model_ready,
        "gpu_ok":          gpu["gpu_ok"],
        "status_stale":    gpu["stale"],
        "status_age_s":    gpu["age_s"],
        "last_error":      gpu["last_error"],
        "last_updated":    gpu["last_updated"],
        "total_requests":  gpu["total_requests"],
        "failed_requests": gpu["failed_requests"],
    }


# ── Prometheus metrics ────────────────────────────────────────────────────────

def _build_metrics() -> str:
    """
    Return Prometheus text.
    Proxies all of Triton's own /metrics output and appends custom gauges
    derived from the model.py status file.
    """
    parts: list[str] = []

    # Triton built-in metrics (request counts, latency histograms, GPU util…)
    triton_metrics = _fetch_triton_metrics()
    if triton_metrics:
        parts.append(triton_metrics.rstrip())
    else:
        parts.append("# WARNING: could not reach Triton metrics endpoint")

    # Custom gauges from the model.py status file
    gpu = _read_gpu_status()
    parts.append("")
    parts.append("# HELP nemo_hybrid_cuda_healthy 1 if GPU context is healthy, 0 if corrupted")
    parts.append("# TYPE nemo_hybrid_cuda_healthy gauge")
    parts.append(f"nemo_hybrid_cuda_healthy {1 if gpu['gpu_ok'] else 0}")

    parts.append("# HELP nemo_hybrid_status_stale 1 if model.py status file is older than STALE_THRESHOLD_S")
    parts.append("# TYPE nemo_hybrid_status_stale gauge")
    parts.append(f"nemo_hybrid_status_stale {1 if gpu['stale'] else 0}")

    parts.append("# HELP nemo_hybrid_status_age_seconds Seconds since model.py last wrote the status file")
    parts.append("# TYPE nemo_hybrid_status_age_seconds gauge")
    parts.append(f"nemo_hybrid_status_age_seconds {gpu['age_s']}")

    parts.append("# HELP nemo_hybrid_total_requests Total inference requests processed")
    parts.append("# TYPE nemo_hybrid_total_requests counter")
    parts.append(f"nemo_hybrid_total_requests {gpu['total_requests']}")

    parts.append("# HELP nemo_hybrid_failed_requests Total inference requests that failed")
    parts.append("# TYPE nemo_hybrid_failed_requests counter")
    parts.append(f"nemo_hybrid_failed_requests {gpu['failed_requests']}")

    # Derived: error rate gauge (NaN-safe)
    total = gpu["total_requests"]
    failed = gpu["failed_requests"]
    error_rate = (failed / total) if total > 0 else 0.0
    parts.append("# HELP nemo_hybrid_error_rate Fraction of requests that failed (0–1)")
    parts.append("# TYPE nemo_hybrid_error_rate gauge")
    parts.append(f"nemo_hybrid_error_rate {error_rate:.6f}")

    parts.append("")
    return "\n".join(parts)


# ── HTTP handler ──────────────────────────────────────────────────────────────

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/health/live":
            state = _liveness_check()
            self._json(200 if state["live"] else 503, state)

        elif path in ("/health/ready", "/health"):
            state = _readiness_check()
            self._json(200 if state["ready"] else 503, state)

        elif path == "/metrics":
            body = _build_metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, code: int, data: dict):
        body = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # silence per-request access logs


# ── CLI (Docker HEALTHCHECK / K8s exec probe) ─────────────────────────────────

def _cli_check(live: bool = False) -> int:
    if live:
        state = _liveness_check()
        ok = state["live"]
        if ok:
            print("OK: live", flush=True)
        else:
            parts = []
            if not state["triton_up"]:
                parts.append("Triton unreachable")
            if not state["gpu_ok"]:
                parts.append(f"GPU unhealthy: {state['last_error']}")
            print(f"UNHEALTHY: {'; '.join(parts)}", flush=True)
        return 0 if ok else 1

    state = _readiness_check()
    ok = state["ready"]
    if ok:
        print(
            f"OK: ready  requests={state['total_requests']}  "
            f"failed={state['failed_requests']}",
            flush=True,
        )
    else:
        parts = []
        if not state["triton_up"]:
            parts.append("Triton unreachable")
        if not state["model_ready"]:
            parts.append("model not loaded")
        if not state["gpu_ok"]:
            parts.append(f"GPU unhealthy: {state['last_error']}")
        if state["status_stale"]:
            parts.append(f"status stale ({state['status_age_s']}s)")
        print(f"UNHEALTHY: {'; '.join(parts)}", flush=True)
    return 0 if ok else 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if "--check-live" in sys.argv:
        sys.exit(_cli_check(live=True))
    if "--check" in sys.argv:
        sys.exit(_cli_check(live=False))

    server = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
    print(
        f"[health_check] Listening on :{HEALTH_PORT}  "
        f"(triton={TRITON_HTTP_URL}  metrics-port={TRITON_METRICS_PORT}  "
        f"status_file={HEALTH_STATUS_FILE})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
