# n1-standard-4 + NVIDIA T4
FROM nvcr.io/nvidia/tensorrt:25.10-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN pip install --upgrade pip \
    && pip install \
        "optimum[onnxruntime-gpu]" \
        transformers \
        torch

COPY export_model.py .
COPY engine_benchmark.py .

CMD ["bash", "-c", "python export_model.py && python engine_benchmark.py"]
