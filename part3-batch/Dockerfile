FROM apache/beam_python3.12_sdk:2.71.0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    pip install \
      "transformers==4.57.6" \
      "onnxruntime==1.24.2"
