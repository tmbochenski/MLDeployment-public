import collections
import json
import os
import statistics
import time

import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Configuration ---
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
NUM_WARMUP_RUNS = 5
NUM_TIMED_RUNS = 20
BATCH_SIZES = [1, 8, 16, 32, 64]
MAX_SEQ_LENGTH = 128


def _safe_model_id(model_name: str) -> str:
    return model_name.replace("/", "_")


def get_benchmark_configs() -> list[dict]:
    safe_name = _safe_model_id(MODEL_NAME)
    fp32_path = f"models/{safe_name}_fp32/model.onnx"
    fp16_path = f"models/{safe_name}_fp16/model_optimized.onnx"

    configs = []
    available_providers = ort.get_available_providers()
    is_cuda_available = torch.cuda.is_available()

    # --- Hugging Face PyTorch Configurations ---
    configs.append(
        {
            "name": "Hugging Face CPU",
            "type": "hf",
            "device": "cpu",
        }
    )
    print("Added: Hugging Face PyTorch (CPU) benchmark.")

    if is_cuda_available:
        configs.append(
            {
                "name": "Hugging Face CUDA",
                "type": "hf",
                "device": "cuda",
            }
        )
        print("Added: Hugging Face PyTorch (CUDA) benchmark.")
    else:
        print("Skipping Hugging Face CUDA benchmark: PyTorch CUDA not available.")

    # --- ONNX Runtime Configurations ---
    if not os.path.exists(fp32_path) or not os.path.exists(fp16_path):
        print("\nWARNING: ONNX models not found. Skipping all ONNX benchmarks.")
        print(f"Please run the conversion script for '{MODEL_NAME}' first.")
        return configs

    configs.append(
        {
            "name": "ONNX CPU (FP32)",
            "type": "onnx",
            "path": fp32_path,
            "provider": ["CPUExecutionProvider"],
        }
    )
    print("Added: ONNX CPU (FP32) benchmark.")

    if "OpenVINOExecutionProvider" in available_providers and False:
        load_config_fp32 = {"CPU": {"NUM_STREAMS": "1", "INFERENCE_NUM_THREADS": "1"}}
        fp32_provider_config = {
            "device_type": "CPU",
            "load_config": json.dumps(load_config_fp32),
        }
        configs.append(
            {
                "name": "ONNX OpenVINO (FP32)",
                "type": "onnx",
                "path": fp32_path,
                "provider": [("OpenVINOExecutionProvider", fp32_provider_config)],
            }
        )

        load_config_bf16 = {"CPU": {"INFERENCE_PRECISION_HINT": "bf16"}}
        bf16_provider_config = {
            "device_type": "CPU",
            "load_config": json.dumps(load_config_bf16),
        }
        configs.append(
            {
                "name": "ONNX OpenVINO (BF16)",
                "type": "onnx",
                "path": fp32_path,
                "provider": [("OpenVINOExecutionProvider", bf16_provider_config)],
            }
        )

        print("Added: ONNX OpenVINO (FP32 & BF16) benchmarks.")
    else:
        print("Skipping OpenVINO benchmarks: 'OpenVINOExecutionProvider' not found.")

    if "CUDAExecutionProvider" in available_providers:
        configs.append(
            {
                "name": "ONNX CUDA (FP32)",
                "type": "onnx",
                "path": fp32_path,
                "provider": ["CUDAExecutionProvider"],
            }
        )
        configs.append(
            {
                "name": "ONNX CUDA (FP16)",
                "type": "onnx",
                "path": fp16_path,
                "provider": ["CUDAExecutionProvider"],
            }
        )
        print("Added: ONNX CUDA (FP32 & FP16) benchmarks.")
    else:
        print("Skipping ONNX CUDA benchmarks: 'CUDAExecutionProvider' not found.")

    if "TensorrtExecutionProvider" in available_providers:
        print(
            "\nNOTE: The first time you run a TensorRT model, it will take several minutes to build an engine."
        )
        trt_provider_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": f"models/trt_cache_{safe_name}",
            "trt_profile_min_shapes": f"input_ids:1x{MAX_SEQ_LENGTH},attention_mask:1x{MAX_SEQ_LENGTH}",
            "trt_profile_opt_shapes": f"input_ids:{max(BATCH_SIZES) // 2}x{MAX_SEQ_LENGTH},attention_mask:{max(BATCH_SIZES) // 2}x{MAX_SEQ_LENGTH}",
            "trt_profile_max_shapes": f"input_ids:{max(BATCH_SIZES)}x{MAX_SEQ_LENGTH},attention_mask:{max(BATCH_SIZES)}x{MAX_SEQ_LENGTH}",
        }
        configs.append(
            {
                "name": "ONNX TensorRT (FP32)",
                "type": "onnx",
                "path": fp32_path,
                "provider": [("TensorrtExecutionProvider", trt_provider_options)],
            }
        )
        configs.append(
            {
                "name": "ONNX TensorRT (FP16)",
                "type": "onnx",
                "path": fp16_path,
                "provider": [("TensorrtExecutionProvider", trt_provider_options)],
            }
        )
        print("Added: ONNX TensorRT (FP32 & FP16) benchmarks.")
    else:
        print("Skipping TensorRT benchmarks: 'TensorrtExecutionProvider' not found.")

    return configs


def _is_cpu_config(config: dict) -> bool:
    if config.get("device") == "cpu":
        return True

    provider_name = ""
    if "provider" in config and config["provider"]:
        provider_info = config["provider"][0]
        provider_name = (
            provider_info if isinstance(provider_info, str) else provider_info[0]
        )

    return provider_name in ["CPUExecutionProvider", "OpenVINOExecutionProvider"]


def run_benchmark(benchmark_configs: list[dict]):
    print(f"\n--- Preparing benchmark data (Batch Size up to {max(BATCH_SIZES)}) ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dummy_text = ["This is a test sentence for benchmarking."] * max(BATCH_SIZES)

    inputs_np = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="np",
    )
    inputs_pt = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    full_data_np = {
        "input_ids": inputs_np["input_ids"],
        "attention_mask": inputs_np["attention_mask"],
    }
    full_data_pt = {
        "input_ids": inputs_pt["input_ids"],
        "attention_mask": inputs_pt["attention_mask"],
    }

    results = collections.defaultdict(dict)

    print("\n--- Starting Inference Benchmark ---")

    for config in benchmark_configs:
        executor_name = config["name"]
        print(f"\nBenchmarking '{executor_name}'...")

        if config["type"] == "hf":
            print(f"  > Model: {MODEL_NAME} (Transformers)")
            print(f"  > Device: {config['device']}")
        elif config["type"] == "onnx":
            provider = config.get("provider")
            provider_name = (
                provider[0] if isinstance(provider[0], str) else provider[0][0]
            )
            print(f"  > Model: {config['path']} (ONNX)")
            print(f"  > Provider: {provider_name}")

        batch_sizes_to_run = BATCH_SIZES
        if _is_cpu_config(config):
            batch_sizes_to_run = BATCH_SIZES[:2]
            print(
                f"  > This is a CPU benchmark. Limiting to batch sizes: {batch_sizes_to_run}"
            )

        # --- Model Setup ---
        if config["type"] == "hf":
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            device = torch.device(config["device"])
            model.to(device)
            model.eval()
        elif config["type"] == "onnx":
            session = ort.InferenceSession(config["path"], providers=config["provider"])
        else:
            raise ValueError(f"Unknown config type: {config['type']}")

        # --- Benchmarking Loop ---
        for batch_size in batch_sizes_to_run:
            if config["type"] == "hf":
                # noinspection PyUnboundLocalVariable
                batch_data = {
                    "input_ids": full_data_pt["input_ids"][:batch_size].to(device),
                    "attention_mask": full_data_pt["attention_mask"][:batch_size].to(
                        device
                    ),
                }
            else:
                batch_data = {
                    "input_ids": full_data_np["input_ids"][:batch_size],
                    "attention_mask": full_data_np["attention_mask"][:batch_size],
                }

            # We measure "latency per batch" (one forward pass over the full batch).
            for _ in range(NUM_WARMUP_RUNS):
                if config["type"] == "hf":
                    with torch.no_grad():
                        # noinspection PyUnboundLocalVariable
                        model(**batch_data)
                else:
                    # noinspection PyUnboundLocalVariable
                    session.run(None, batch_data)

            durations = []
            for _ in range(NUM_TIMED_RUNS):
                if config["type"] == "hf" and device.type == "cuda":
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                if config["type"] == "hf":
                    with torch.no_grad():
                        model(**batch_data)
                else:
                    session.run(None, batch_data)
                end_time = time.perf_counter()

                if config["type"] == "hf" and device.type == "cuda":
                    torch.cuda.synchronize()

                durations.append(end_time - start_time)

            results[batch_size][executor_name] = durations
            print(
                f"  > Completed Batch Size {batch_size}: "
                f"{statistics.mean(durations) * 1000:.2f} ms/batch"
            )

    return results


def print_results(results: dict):
    print("\n--- Benchmark Results ---")
    for batch_size, model_timings in sorted(results.items()):
        print(f"\nBatch Size: {batch_size}")
        header = f"{'Executor':<25} | {'Avg Latency (ms)':<20} | {'Std Dev (ms)':<15} | {'Throughput (IPS)':<20}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for executor_name, durations in sorted(model_timings.items()):
            avg_latency_ms = statistics.mean(durations) * 1000
            std_dev_ms = (
                statistics.stdev(durations) * 1000 if len(durations) > 1 else 0.0
            )
            throughput_ips = batch_size / statistics.mean(durations)

            print(
                f"{executor_name:<25} | {avg_latency_ms:<20.3f} | {std_dev_ms:<15.3f} | {throughput_ips:<20.2f}"
            )
        print("-" * len(header))


def main() -> None:
    print("=== Benchmark run starting ===")
    print(f"Model id: {MODEL_NAME}")

    benchmark_configurations = get_benchmark_configs()
    all_results = run_benchmark(benchmark_configurations)
    print_results(all_results)
    print("\n--- Benchmark Finished ---")


if __name__ == "__main__":
    main()
