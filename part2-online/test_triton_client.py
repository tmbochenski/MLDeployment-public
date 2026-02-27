import argparse
import subprocess
import time

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


MODELS = ["resnet50", "twitter-sentiment"]


def _get_headers(use_gcloud: bool) -> dict[str, str]:
    if use_gcloud:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-identity-token"],
            text=True,
        ).strip()
        return {"Authorization": f"Bearer {token}", "Connection": "close", "Content-Type": "application/json"}

    return {"Connection": "close", "Content-Type": "application/json"}


def _build_io_for_model(
    model_name: str,
) -> tuple[list[httpclient.InferInput], list[httpclient.InferRequestedOutput]]:
    if model_name == "resnet50":
        batch_size = 1
        input_data = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
        inputs = [
            httpclient.InferInput(
                name="input__0", shape=list(input_data.shape), datatype="FP32"
            )
        ]
        inputs[0].set_data_from_numpy(input_data, binary_data=False)
        outputs = [httpclient.InferRequestedOutput(name="resnet50_output")]
        return inputs, outputs

    if model_name == "twitter-sentiment":
        texts = ["I love ChatGPT!", "This is terrible."]
        input_data = np.array(texts, dtype=object)
        inputs = [
            httpclient.InferInput(
                name="input__0", shape=list(input_data.shape), datatype="BYTES"
            )
        ]
        inputs[0].set_data_from_numpy(input_data, binary_data=False)
        outputs = [httpclient.InferRequestedOutput(name="twitter_sentiment_output")]
        return inputs, outputs

    raise ValueError(
        f"Unknown model {model_name!r}. "
        f"Add it to _build_io_for_model() if you want to test it."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Triton models (local or Cloud Run)."
    )
    parser.add_argument(
        "--url", default="localhost:8080", help="Triton host:port (no scheme)."
    )
    parser.add_argument(
        "--ssl", action="store_true", help="Use SSL/TLS (needed for Cloud Run)."
    )
    parser.add_argument("--auth-gcloud", action="store_true", help="Auth using gcloud.")
    args = parser.parse_args()

    headers = _get_headers(args.auth_gcloud)

    client = httpclient.InferenceServerClient(url=args.url, ssl=args.ssl)

    print("=== OIP Multi-Model Server Test (Triton Client) ===\n")
    print("URL:", args.url)
    print("SSL:", args.ssl)
    print("Auth gcloud:", args.auth_gcloud, "\n")

    print("Server live:", client.is_server_live(headers=headers))
    print("Server ready:", client.is_server_ready(headers=headers), "\n")

    for model_name in MODELS:
        print(f"Testing model: {model_name}")
        print("  Model ready:", client.is_model_ready(model_name, headers=headers))

        inputs, outputs = _build_io_for_model(model_name)

        start = time.perf_counter()
        try:
            result = client.infer(
                model_name, inputs=inputs, outputs=outputs, headers=headers
            )
            latency_ms = (time.perf_counter() - start) * 1000
            print(f"  Latency: {latency_ms:.2f} ms")
            print("  Outputs:")

            for output in outputs:
                output_name = output.name()
                output_data = result.get_output(output_name)
                print(f"    {output_name}: {output_data}\n")

        except InferenceServerException as e:
            print(f"  Error during {model_name} inference:", e, "\n")


if __name__ == "__main__":
    main()
