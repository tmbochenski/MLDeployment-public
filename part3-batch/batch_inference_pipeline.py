import argparse
import csv
import io
from typing import Iterable
import json
import os
import hashlib

import numpy as np
import onnxruntime as ort
import transformers
import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions


INPUT_FILE = "gs://dataflow-ml-deployment-488023/input/user_posts.csv"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TOP_N = 5

ONNX_MODEL_URI = (
    "gs://dataflow-ml-deployment-488023/models/"
    "cardiffnlp_twitter-roberta-base-sentiment-latest_fp32/model.onnx"
)

USERS_OF_INTEREST = ["user_0007", "user_0428", "user_0815"]

OUTPUT_HIGH_PRIORITY_ALERTS = "high_priority_negative_alerts.json"
OUTPUT_USERS_OF_INTEREST_METRICS = "users_of_interest_metrics.json"
OUTPUT_ALL_PREDICTIONS = "all_predictions.json"
OUTPUT_TOP_POSITIVE_POSTS = f"top_{TOP_N}_positive_posts.json"
OUTPUT_TOP_NEGATIVE_POSTS = f"top_{TOP_N}_negative_posts.json"


class ParsePostsCsvLineDoFn(beam.DoFn):
    def __init__(self, fieldnames: list[str]):
        super().__init__()
        self._fieldnames = fieldnames

    def process(self, line: str, **kwargs) -> Iterable[dict]:
        # Parse exactly one CSV record from this line.
        reader = csv.DictReader(io.StringIO(line), fieldnames=self._fieldnames)
        row = next(reader)

        post_id = row["post_id"].strip()
        user_id = row["user_id"].strip()
        post_text = row["text"]

        yield {"post_id": post_id, "user_id": user_id, "text": post_text}


class TopNCombineFn(beam.CombineFn):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def create_accumulator(self):
        return []

    def add_input(self, accumulator, element, **kwargs):
        accumulator.append(element)
        accumulator.sort(key=lambda post: post["confidence"], reverse=True)
        return accumulator[: self.n]

    def merge_accumulators(self, accumulators, **kwargs):
        combined_list = [item for sublist in accumulators for item in sublist]
        combined_list.sort(key=lambda post: post["confidence"], reverse=True)
        return combined_list[: self.n]

    def extract_output(self, accumulator, **kwargs):
        return accumulator


class InferAndClassifyParDo(beam.DoFn):
    TAG_HIGH_PRIORITY_ALERTS = "high_priority_alerts"

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.config = None
        self.session = None
        self.input_names = None
        self.output_name = None

    def setup(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
        self.config = transformers.AutoConfig.from_pretrained(MODEL_NAME)

        # First run downloads the model; next runs reuse the cached file.
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mldeployment")
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = (
            hashlib.sha256(ONNX_MODEL_URI.encode("utf-8")).hexdigest()[:16] + ".onnx"
        )
        cached_onnx_path = os.path.join(cache_dir, cache_name)

        if not os.path.exists(cached_onnx_path):
            print(f"Downloading ONNX model once to: {cached_onnx_path}")
            with (
                FileSystems.open(ONNX_MODEL_URI) as src,
                open(cached_onnx_path, "wb") as dst,
            ):
                dst.write(src.read())

        print("Loading ONNX model...")
        self.session = ort.InferenceSession(
            cached_onnx_path,
            providers=["CPUExecutionProvider"],
        )
        print("ONNX model loaded.")

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

    def process(self, batch: list[dict], **kwargs):
        # <IMPLEMENT ME>


class CalculateUserSentimentScore(beam.DoFn):
    def process(self, element, **kwargs):
        # <IMPLEMENT ME>


def run_pipeline(pipeline_options: PipelineOptions):
    p = beam.Pipeline(options=pipeline_options)

    fieldnames = ["post_id", "user_id", "text"]

    posts = (
        p
        | "ReadCSVLines" >> beam.io.ReadFromText(INPUT_FILE, skip_header_lines=1)
        | "ParseCSVLines" >> beam.ParDo(ParsePostsCsvLineDoFn(fieldnames=fieldnames))  # type: ignore[arg-type]
    )

    # <IMPLEMENT ME>

    return p.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wait",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Wait for the Beam pipeline to finish (use --no-wait to return immediately).",
    )
    args, pipeline_args = parser.parse_known_args()

    options = PipelineOptions(pipeline_args)
    options.view_as(SetupOptions).save_main_session = True

    result = run_pipeline(options)
    if args.wait:
        result.wait_until_finish()
