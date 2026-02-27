# n4d-standard-4
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        openvino \
        "optimum[onnxruntime]" \
        onnxruntime-openvino \
        transformers && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch

COPY export_model.py .
COPY engine_benchmark.py .

CMD ["bash", "-c", "python export_model.py && python engine_benchmark.py"]
