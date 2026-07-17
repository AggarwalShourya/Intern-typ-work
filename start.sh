#!/bin/bash
set -e

echo "MODEL_PATH:       ${MODEL_PATH}"
echo "USE_TRT_ENCODER:  ${USE_TRT_ENCODER:-false}"

MODEL_DIR="/models/nemo_hybrid"
MODEL_FILE="${MODEL_DIR}/albatrossv1-beta.nemo"
TRT_ENGINE_LOCAL="/trt_cache/encoder_fp16.trt"
LANG_MAP_FILE="${MODEL_DIR}/lang_to_tokens.json"

# ── Download .nemo weights ────────────────────────────────────────────────────
if [ ! -f "${MODEL_FILE}" ]; then
    echo "Downloading NeMo model from GCS..."
    gsutil -m cp "${MODEL_PATH}" "${MODEL_FILE}"
else
    echo "NeMo model already present, skipping download."
fi

# ── Generate language token map ───────────────────────────────────────────────
# Loads tokenizer directly from the NeMo model and classifies every BPE token
# by Unicode script into {lang: [token_ids]}.  Cached after first run.
if [ ! -f "${LANG_MAP_FILE}" ]; then
    echo "Generating language token map from NeMo model tokenizer..."
    python3 /workspace/create_lang_token_map.py \
        --nemo-model "${MODEL_FILE}" \
        --output-dir "${MODEL_DIR}"
    echo "Language token map generated at ${LANG_MAP_FILE}"
else
    echo "Language token map already present, skipping generation."
fi

export LANG_TOKEN_MAP_PATH="${LANG_MAP_FILE}"
echo "LANG_TOKEN_MAP_PATH=${LANG_TOKEN_MAP_PATH}"

# ── Build TRT FP16 encoder engine (optional) ─────────────────────────────────
# Set USE_TRT_ENCODER=true to enable.  The engine is built once and cached at
# TRT_ENGINE_LOCAL; subsequent container restarts skip the build entirely.
#
# Shape tuning via env vars (all optional — see build_trt_encoder.py):
#   TRT_MIN_BATCH, TRT_OPT_BATCH, TRT_MAX_BATCH
#   TRT_MIN_FRAMES, TRT_OPT_FRAMES, TRT_MAX_FRAMES

if [ "${USE_TRT_ENCODER}" = "true" ]; then
    mkdir -p "$(dirname "${TRT_ENGINE_LOCAL}")"

    # Validate any existing engine — a failed/partial build leaves a corrupt file.
    if [ -f "${TRT_ENGINE_LOCAL}" ]; then
        TRT_ENGINE_LOCAL="${TRT_ENGINE_LOCAL}" python3 - <<'PYEOF'
import sys, os, tensorrt as trt
engine_path = os.environ["TRT_ENGINE_LOCAL"]
runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
with open(engine_path, "rb") as fh:
    engine = runtime.deserialize_cuda_engine(fh.read())
if engine is None:
    print(f"[start.sh] Cached TRT engine is corrupt — removing and rebuilding.", flush=True)
    os.remove(engine_path)
    sys.exit(1)
print(f"[start.sh] Cached TRT engine OK ({engine.num_io_tensors} tensors).", flush=True)
PYEOF
    fi

    if [ ! -f "${TRT_ENGINE_LOCAL}" ]; then
        echo "TRT engine not found — building now (takes 5-15 minutes) ..."
        python3 /workspace/build_trt_encoder.py \
            --nemo   "${MODEL_FILE}" \
            --engine "${TRT_ENGINE_LOCAL}"
        echo "TRT engine build complete."
    else
        echo "TRT engine already present, skipping build."
    fi

    export TRT_ENCODER_PATH="${TRT_ENGINE_LOCAL}"
    echo "TRT_ENCODER_PATH=${TRT_ENCODER_PATH}"
else
    echo "USE_TRT_ENCODER not set to 'true' — using PyTorch encoder."
fi

# ── Start health-check sidecar (background) ─────────────────────────────────
# Serves four endpoints on HEALTH_PORT (default 9005):
#   /health/live   — liveness probe  (Triton up + GPU not corrupted)
#   /health/ready  — readiness probe (model loaded + GPU healthy + not stale)
#   /health        — full JSON status (alias for /health/ready)
#   /metrics       — Prometheus: proxies Triton :9003/metrics + GPU gauges
echo "Starting health-check server on port ${HEALTH_PORT:-9005}..."
python3 /workspace/health_check.py &

echo "Starting Triton Inference Server..."
exec tritonserver \
    --model-repository=/models \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002 \
    --exit-on-error=false \
    --log-verbose=1 \
    --model-load-thread-count=1 \
    --cache-config=local,size=67108864 \
    --cuda-memory-pool-byte-size=0:2147483648 \
    --pinned-memory-pool-byte-size=536870912 \
    --backend-config=python,stub-timeout-seconds=600
