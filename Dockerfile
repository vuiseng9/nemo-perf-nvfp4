FROM nvcr.io/nvidia/nemo:25.09
SHELL ["/bin/bash", "-lc"]
USER root

# --- Configurable args ---
ARG TE_WHEEL_URL="https://huggingface.co/vuiseng9/te-wheel/resolve/main/unofficial/251025_a/transformer_engine-2.8.0%2B7cde9a33-cp312-cp312-linux_x86_64.whl"
ARG NEMO_REF="r2.5.0-perf-local"
ARG MEGATRON_REF="core_v0.14.0+nvfp4"

# Hardcode Python 3.12 site-packages path
ENV PY_SITE="/usr/local/lib/python3.12/dist-packages"

# --- System deps ---
RUN set -euo pipefail \
    && apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates ffmpeg tree graphviz \
    && rm -rf /var/lib/apt/lists/*

# --- Activate venv ---
RUN source /opt/venv/bin/activate && python -V

# --- Transformer Engine wheel install ---
ENV PIP_NO_CACHE_DIR=1
RUN python3 -m pip install --no-cache-dir "${TE_WHEEL_URL}"

# --- Replace Megatron-LM under /opt with your fork/branch ---
WORKDIR /opt
RUN set -euo pipefail \
    && if [ -d "megatron-lm" ]; then mv megatron-lm "old.megatron-lm"; fi \
    && git clone https://github.com/vuiseng9/megatron-lm \
    && cd megatron-lm \
    && git fetch --all --tags \
    && git checkout "${MEGATRON_REF}" \
    && python3 -m pip install --no-build-isolation --no-cache-dir .

WORKDIR /opt/NeMo
RUN set -euo pipefail \
    && git remote add fork https://github.com/vuiseng9/NeMo \
    && git fetch fork \
    && git checkout "${NEMO_REF}"

WORKDIR /workspace
