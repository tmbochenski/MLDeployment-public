from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig
import uvicorn
import os
from typing import Any
from contextlib import asynccontextmanager


HF_TWITTER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ONNX_TWITTER_MODEL = (
    "models/cardiffnlp_twitter-roberta-base-sentiment-latest_fp32/model.onnx"
)


@asynccontextmanager
async def lifespan(fast_app: FastAPI):
    fast_app.state.models = {
        "resnet50": DummyImageModel(),
        "twitter-sentiment": TwitterSentimentONNXModel(),
    }
    yield


app = FastAPI(title="OIP Multi-Model Server (ONNX)", lifespan=lifespan)


class InputTensor(BaseModel):
    # todo


class InferenceRequest(BaseModel):
    # todo


class InferenceOutput(BaseModel):
    # todo


class InferenceResponse(BaseModel):
    # todo


class DummyImageModel:
    name = "resnet50"

    def __init__(self):
        self.ready = False
        self.ready = True

    # noinspection PyMethodMayBeStatic
    def infer(self, inputs: list[InputTensor]) -> list[InferenceOutput]:
        input_tensor = inputs[0]
        data = np.array(input_tensor.data, dtype=np.float32)
        data = data.reshape(input_tensor.shape)
        batch_size = data.shape[0]
        out = np.random.rand(batch_size, 1000).tolist()
        return [
            InferenceOutput(
                name="resnet50_output",
                datatype="FP32",
                shape=[batch_size, 1000],
                data=out,
            )
        ]


class TwitterSentimentONNXModel:
    name = "twitter-sentiment"

    def __init__(self, model_path: str = ONNX_TWITTER_MODEL):
        self.ready = False
        self.tokenizer = AutoTokenizer.from_pretrained(HF_TWITTER_MODEL)
        self.config = AutoConfig.from_pretrained(HF_TWITTER_MODEL)
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.attention_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name
        self.ready = True

    def infer(self, inputs: list[InputTensor]) -> list[InferenceOutput]:
        # todo


@app.get("/v2/health/live")
async def liveness():
    # todo


@app.get("/v2/health/ready")
async def readiness(request: Request):
    models = request.app.state.models
    # todo


@app.get("/v2/models/{model_name}/ready")
async def model_readiness(model_name: str, request: Request):
    models = request.app.state.models
    # todo


@app.post("/v2/models/{model_name}/infer", response_model=InferenceResponse)
async def infer(model_name: str, request: InferenceRequest, http_request: Request):
    models = http_request.app.state.models
    # todo


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("oip_multi_model_server:app", host=host, port=port, reload=False)
