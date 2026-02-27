FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        pydantic \
        numpy \
        onnxruntime \
        transformers

COPY oip_multi_model_server.py .
COPY models/ ./models/

EXPOSE 8080

ENV HF_HUB_DISABLE_SSL_VERIFICATION=1

CMD ["uvicorn", "oip_multi_model_server:app", "--host", "0.0.0.0", "--port", "8080"]
